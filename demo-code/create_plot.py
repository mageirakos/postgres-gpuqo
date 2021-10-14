import toytree
import toyplot
import toyplot.svg

canvas, axes, mark =  toytree.tree("( (((R1, R2), (R3, (R12,(R13,R14)))), ((R4,(R15,(R16,(R17,R18)))),R5))  ,((R6,(R9,(R7,R10))), (R8,(R11,(R19,R20))) )  );").draw(layout='d',
                                                                    width=400,
                                                                    height=300,
                                                                    node_hover=True,
                                                                    node_labels="⋈",
                                                                    node_sizes=10,
                                                                    tip_labels_align=True,
                                                                    edge_type='c',)
toytree.tree("(((R1, R2), R3),(R4,R5));").draw(layout='d',
                            node_hover=True,
                                     node_labels="⋈",
                                     node_sizes=15,
                                     tip_labels_align=True,
                                     edge_type='c',
                                    )

toyplot.svg.render(canvas, "./tree-plot.svg")
