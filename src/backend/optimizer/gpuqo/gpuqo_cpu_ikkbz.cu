/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_ikkbz.cu
 *      IKKBZ approximate algorithm
 *
 * src/backend/optimizer/gpuqo/gpuqo_ikkbz.cu
 *
 *-------------------------------------------------------------------------
 */

#include <stack>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_common.cuh"
#include "gpuqo_query_tree.cuh"

using namespace std;

template<typename BitmapsetN>
class IKKBZRel {
    BitmapsetN relid;
    float sel;
    float rows;

public:
    IKKBZRel<BitmapsetN>(const BaseRelation<BitmapsetN> &baserel){
        relid = baserel.id;
        rows = baserel.rows;
        sel = NANF;
    }

    void setSel(float _sel) {
        sel = _sel;
    }

    float getSel() const {
        return sel;
    }

    float getRows() const {
        return rows;
    }

    BitmapsetN getRelid() const {
        return relid;
    }

    int getIdx() const {
        Assert(relid.size() == 1);
        return relid.lowestPos()-1;
    }

    float getSelRows() const {
        return sel * rows;
    }
};

template<typename BitmapsetN>
using rel_list_t = list<const IKKBZRel<BitmapsetN>*>;

template<typename BitmapsetN>
class IKKBZNode {
    rel_list_t<BitmapsetN> rels;
    float T;
    float C;
    IKKBZNode<BitmapsetN> *child;
    IKKBZNode<BitmapsetN> *sibling;

public:
    IKKBZNode<BitmapsetN>(const IKKBZRel<BitmapsetN> *rel, float _T, float _C) :
           T(_T), C(_C), child(NULL), sibling(NULL) 
    {
        rels.push_back(rel);
    }

    void merge()
    {
        Assert(hasOneChild());

        C += T * child->C;
        T *= child->T;

        for (auto r:child->rels)
            rels.push_back(r);

        IKKBZNode<BitmapsetN> *tmp = child;
        child = child->child;
        delete tmp;
    }
    
    IKKBZNode<BitmapsetN> *getChild() const {
        return child;
    }

    bool hasChildren() const {
        return child != NULL;
    }

    bool hasSiblings() const {
        return sibling != NULL;
    }

    bool hasOneChild() const {
        return child != NULL && child->sibling == NULL;
    }
    
    IKKBZNode<BitmapsetN> *getSibling() const {
        return sibling;
    }
    
    void setChild(IKKBZNode<BitmapsetN> *_child) {
        child = _child;
    }
    
    void setSibling(IKKBZNode<BitmapsetN> *_sibling) {
        sibling = _sibling;
    }

    float rank() const {
        return C ? (T-1)/C : 0.0f;
    }

    int getBaseRelIdx() const {
        Assert(rels.size() == 1);
        return rels.back()->getIdx();
    }

    const rel_list_t<BitmapsetN> &getRels() const {
        return rels;
    }

    void addChild(IKKBZNode<BitmapsetN> *other) {
        IKKBZNode<BitmapsetN> *node = child, *prev = NULL;
        
        while (node != NULL){
            prev = node;
            node = node->sibling;
        }
        
        if (!prev) {
            child = other;
        } else {
            prev->sibling = other;
        }
    }

    BitmapsetN getRelid() const {
        BitmapsetN out(0);

        for (auto rel:rels) {
            out |= rel->getRelid();
        }
        
        return out;
    }

    float getCost() const {
        return C;
    }
};

template<typename BitmapsetN>
static
IKKBZNode<BitmapsetN> *ikkbz_make_tree(int root_idx, 
                                    vector<IKKBZRel<BitmapsetN>*> &rels,
                                    GpuqoPlannerInfo<BitmapsetN> *info)
{
    typedef pair<IKKBZNode<BitmapsetN>*, IKKBZNode<BitmapsetN>*> stack_el_t;

    IKKBZNode<BitmapsetN> *root = new IKKBZNode<BitmapsetN>(
        rels[root_idx], rels[root_idx]->getRows(), 0.0f
    );

    stack<stack_el_t> dfs_stack;
    dfs_stack.push(make_pair(root, (IKKBZNode<BitmapsetN>*)NULL));

    while (!dfs_stack.empty()){
        stack_el_t p = dfs_stack.top();
        IKKBZNode<BitmapsetN> *node = p.first;
        IKKBZNode<BitmapsetN> *parent = p.second;

        dfs_stack.pop();

        BitmapsetN edges = info->edge_table[node->getBaseRelIdx()];
        while (!edges.empty()) {
            int neig_baserel_idx = edges.lowestPos()-1;
            IKKBZRel<BitmapsetN> *neig_rel = rels[neig_baserel_idx];

            if (!parent || parent->getBaseRelIdx() != neig_baserel_idx) {          
                rels[neig_baserel_idx]->setSel(estimate_join_selectivity(
                    node->getRelid(), neig_rel->getRelid(), info
                ));

                IKKBZNode<BitmapsetN> *new_node = new IKKBZNode<BitmapsetN>(
                    neig_rel, neig_rel->getSelRows(), neig_rel->getSelRows()
                );

                node->addChild(new_node);

                dfs_stack.push(make_pair(new_node, node));
            }

            edges.unset(neig_baserel_idx+1);
        }
    }

    return root;
}

template<typename BitmapsetN>
static IKKBZNode<BitmapsetN>*
find_node_with_chain_children(IKKBZNode<BitmapsetN> *node) 
{
    if (!node->hasChildren()) {
        return NULL;
    } else if (node->hasOneChild()) {
        return find_node_with_chain_children(node->getChild());
    } else {
        IKKBZNode<BitmapsetN> *p = node->getChild();
        
        while(p) {
            IKKBZNode<BitmapsetN> *ret = find_node_with_chain_children(p);
            if (ret) {
                return ret;
            }
            p = p->getSibling();
        }

        return node;
    }
}

template<typename BitmapsetN>
static void
IKKBZ_normalize(IKKBZNode<BitmapsetN> *root) 
{
    IKKBZNode<BitmapsetN> *node = root, *prev = NULL;

    while (node) {
        if (prev) {
            Assert(!node->hasSiblings());
            if (prev->rank() > node->rank()) {
                prev->merge();
                node = prev->getChild();
                continue;
            }
        }
        prev = node;
        node = node->getChild();
    }
}

template<typename BitmapsetN>
static void
IKKBZ_merge_children(IKKBZNode<BitmapsetN> *parent)
{
    IKKBZNode<BitmapsetN> *root = NULL, *node = NULL;

    while(parent->getChild()) {
        IKKBZNode<BitmapsetN> *min_chain = NULL, *prev_min_chain = NULL;
        IKKBZNode<BitmapsetN> *it = parent->getChild(), *prev = NULL;

        while(it) 
        {
            if (!min_chain || it->rank() < min_chain->rank()) {
                min_chain = it;
                prev_min_chain = prev;
            }

            prev = it;
            it = it->getSibling();
        }

        if (node) {
            node->setChild(min_chain);
            node->setSibling(NULL);
        } else {
            root = min_chain;
        }

        node = min_chain;

        if (min_chain->hasChildren()) {
            Assert(min_chain->hasOneChild());
            min_chain->getChild()->setSibling(min_chain->getSibling());
            if (prev_min_chain) {
                prev_min_chain->setSibling(min_chain->getChild());
            } else {
                parent->setChild(min_chain->getChild());
            }
        } else {
            if (prev_min_chain) {
                prev_min_chain->setSibling(min_chain->getSibling());
            } else {
                parent->setChild(min_chain->getSibling());
            }
        }   
    }
    parent->setChild(root);
}

template<typename BitmapsetN>
static IKKBZNode<BitmapsetN>*
chain_to_idxlist(IKKBZNode<BitmapsetN>* chain, list<int> &idxs)
{
    while (chain) {
        for (auto rel:chain->getRels()){
            idxs.push_back(rel->getIdx());
        }
        chain = chain->getChild();
    }
}

template<typename BitmapsetN>
static IKKBZNode<BitmapsetN>*
IKKBZ_iter(int root_idx, vector<IKKBZRel<BitmapsetN>*> &rels,
            GpuqoPlannerInfo<BitmapsetN> *info)
{
    IKKBZNode<BitmapsetN> *v_prime, *v = ikkbz_make_tree(root_idx, rels, info);

    while (v_prime = find_node_with_chain_children(v)){
        IKKBZNode<BitmapsetN> *node = v_prime->getChild();
        while (node) {
            IKKBZ_normalize(node);
            node = node->getSibling();
        }

        IKKBZ_merge_children(v_prime);
    }

    return v;
}

template<typename BitmapsetN>
static QueryTree<BitmapsetN>*
make_query(IKKBZNode<BitmapsetN>* chain, 
            GpuqoPlannerInfo<BitmapsetN> *info)
{
    IKKBZNode<BitmapsetN> *node = chain;
    JoinRelationCPU<BitmapsetN> *jr = NULL;
    QueryTree<BitmapsetN> *out;

    while (node) {
        for (auto rel:node->getRels()) {
            int i = rel->getIdx();
            JoinRelationCPU<BitmapsetN> *bjr = new JoinRelationCPU<BitmapsetN>;
            bjr->id = info->base_rels[i].id; 
            bjr->left_rel_id = 0; 
            bjr->left_rel_ptr = NULL; 
            bjr->right_rel_id = 0; 
            bjr->right_rel_ptr = NULL; 
            bjr->cost = cost_baserel(info->base_rels[i]); 
            bjr->width = info->base_rels[i].width; 
            bjr->rows = info->base_rels[i].rows; 
            bjr->edges = info->edge_table[i];

            if (jr) {
                Assert(are_valid_pair(jr->id, bjr->id, info));
                jr = make_join_relation(*jr, *bjr, info);
            } else {
                jr = bjr;
            }
        }
        
        node = node->getChild();
    }

    build_query_tree(jr, &out);

    JoinRelationCPU__free(jr);

    return out;
}

/* gpuqo_cpu_ikkbz
 *
 *	 IKKBZ approximation
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_ikkbz(GpuqoPlannerInfo<BitmapsetN> *orig_info)
{
    IKKBZNode<BitmapsetN> *best = NULL; 

    GpuqoPlannerInfo<BitmapsetN> *info = minimumSpanningTree(orig_info);

    vector<IKKBZRel<BitmapsetN>*> rels(info->n_rels);
    for (int i = 0; i < info->n_rels; i++)
        rels[i] = new IKKBZRel<BitmapsetN>(info->base_rels[i]);

    for (int v_id = 0; v_id < info->n_rels; v_id++) {
        IKKBZNode<BitmapsetN> *v = IKKBZ_iter(v_id, rels, info);

        while(v->hasChildren()){
            v->merge();
        }

        LOG_DEBUG("root: %d, cost: %.2f\n", v_id, v->getCost());

        if (!best || v->getCost() < best->getCost()) {
            if (best)
                delete best;
            best = v;
        } else {
            delete v;
        }
    }

    QueryTree<BitmapsetN> *qt = make_query(best, info);

    for (int i = 0; i < info->n_rels; i++)
        delete rels[i];

    delete best;
    freeGpuqoPlannerInfo(info);

    return qt;

}

template QueryTree<Bitmapset32>* gpuqo_cpu_ikkbz<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_ikkbz<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic>* gpuqo_cpu_ikkbz<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>*);
