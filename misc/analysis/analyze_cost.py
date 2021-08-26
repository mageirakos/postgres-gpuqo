#!/bin/env python3

import argparse
import os
from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T
from scipy.stats.mstats import gmean
from itertools import cycle, chain
from collections import defaultdict
from math import inf
import csv

def folder2series(folder, depth=1):
    return '/'.join(folder.split('/')[-depth:]) # remove final slash here or on command line 

def ratio(v, ref):
    return v/ref

def count_tables(query):
    return len(query[query.find("FROM"):query.find("WHERE")].split(','))

def read_query(filename):
    with open(filename) as f:
        lines = f.readlines()
        s = ' '.join(l.strip() for l in lines)
        return s


def load_results(folder):
    queries = {}
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as f:
            query = None
            
            for line in chain(f.readlines(), ['end.sql\n']):
                if 'sql' in line:
                    # if query and queries[query].get('plan_time') is None:
                    #     del queries[query]

                    query = line[:-5]

                    if query != 'end':
                        queries[query] = {}
                elif any(
                    s in line 
                    for s in ['gpuqo_cost','gpuqo_time', 
                              'plan_time', 'postgres_cost']
                ):
                    sp = line.split()
                    key = sp[0]
                    val = float(sp[1]) if len(sp) > 1 else None

                    if query and query in queries:
                        queries[query][key] = val

    return queries

def add_table_count(queries, sql_folder):
    for query, d in queries.items():
        if sql_folder:
            d['tables'] = count_tables(
                read_query(
                    os.path.join(sql_folder, os.path.basename(query + '.sql'))
                    if sql_folder else query + '.sql'
                    )
                )
        else:
            d['tables'] = int(query.split('/')[-1][:-2])
    return queries

def remove_incomplete_runs(queries, metric):
    time_metric = "plan_time" if metric == "postgres_cost" else "gpuqo_time"
    min_crashing_tables = np.inf
    for query, d in queries.items():
        if d.get(time_metric) is None:
            print("WARN:", query, "is incomplete")
            min_crashing_tables = min(
                min_crashing_tables,
                d['tables']
            )
    
    remove_keys = set()
    for query, d in queries.items():
        if d["tables"] >= min_crashing_tables:
            remove_keys.add(query)
        
    for key in remove_keys:
        del queries[key]

def load_results_complete(result_folder, sql_folder, metric):
    queries = load_results(result_folder)
    add_table_count(queries, sql_folder)
    remove_incomplete_runs(queries, metric)
    return queries

def scatter_plot_count_cost(queries, metric, label=None, ratio=False, color=None, marker="o", shift=0, marker_size=100):
    keys = sorted(list(queries.keys()))
    x = np.array([queries[k]['tables'] for k in keys]) + shift
    y = [queries[k][metric] for k in keys]
    plt.errorbar(x, y, linestyle="None", color=color)

    plt.scatter(x, y, label=label, color=color, marker=marker, s=marker_size)

def line_plot_count_cost(queries, metric, label=None, ratio=False, color=None, errorbar=False):
    keys = sorted(list(queries.keys()))
    x = np.array([queries[k]['tables'] for k in keys])
    x_line = []
    y_line = []
    y_err = []

    if ratio:
        y = np.array([queries[k][metric] for k in keys])

        for n in np.sort(np.unique(x)):
            mask = (x == n) & (y > 0)
            if np.any(mask):
                x_line.append(n)
                y_line.append(gmean(y[mask]))   
                y_err.append([np.min(y[mask]), np.max(y[mask])]) 
    else:
        y = np.array([queries[k][metric] for k in keys])

        for n in np.sort(np.unique(x)):
            mask = (x == n) & (y > 0)
            
            if np.any(mask):
                x_line.append(n)
                y_line.append(np.average(y[mask]))   
                y_err.append([np.min(y[mask]), np.max(y[mask])])

    if errorbar:
        plt.errorbar(x_line, y_line, yerr=np.array(y_err).T, label=label, color=color)
    else:
        plt.plot(x_line, y_line, label=label, color=color)

def generic_plot(series, line=False, scatter=True, 
                metric="postgres_cost_ratio", max_shift=0):
    markers = cycle(['+','x','1','2','3','4',])
    shifts = np.linspace(-max_shift, max_shift, len(series))
    for i, (s, queries) in enumerate(series.items()):
        if scatter:
            scatter_plot_count_cost(
                queries, 
                metric=metric, 
                label=s if not line else None, 
                ratio='ratio' in metric,
                color=f"C{i}", 
                marker=next(markers),
                shift=shifts[i],
                marker_size=10 if line else 40
            )
        if line: 
            line_plot_count_cost(
                queries, 
                metric=metric, 
                label=s, 
                ratio='ratio' in metric,
                color=f"C{i}",
                errorbar=(not scatter)
            )

    # configure matplotlib
    if 'ratio' in metric:
        plt.ylabel("Cost ratio (lower is better)")
        plt.yscale("log")
    else:
        plt.ylabel("Cost")

    plt.legend()
    plt.grid(which="major")
    plt.grid(which="minor", linestyle="--")
    plt.xlabel("Number of tables")
    plt.autoscale(True, 'both', True)


def scatter_plot(*args, **kwargs):
    return generic_plot(*args, **kwargs, line=False, scatter=True)

def line_plot(*args, **kwargs):
    return generic_plot(*args, **kwargs, line=True, scatter=False)

def scatter_line_plot(*args, **kwargs):
    return generic_plot(*args, **kwargs, line=True, scatter=True)

def export_csv_aggr(csv_file, series, metric):
    sizes_set = set()
    for _, queries in series.items():
        for _, d in queries.items():
            sizes_set.add(d['tables'])

    sizes = sorted(sizes_set)
    header = (
        "algorithm",
        *(f"{s} ({t})" for s in sizes for t in ['avg', '95%'])
    )
    output = []
    
    float_formatter = "{:.2f}".format
    # float_formatter = "{}".format
    for s, queries in series.items():
        out = [s]
        x = np.array([queries[k]['tables'] for k in queries])
        y = np.array([queries[k][metric] 
                        if queries[k][metric] is not None 
                        else 0 
                    for k in queries])
        for n in sizes:
            mask = (x == n) & (y > 0)
            if np.any(mask):
                out.append(float_formatter(np.average(y[mask])))
                # print(out," ")
                out.append(float_formatter(np.percentile(y[mask], 95)))
            else:
                out.append(None)
                out.append(None)
            
        output.append(tuple(out))

    # print(f" HERE is the output: {output}")
    # print(f"type: {type(output)}")
    # for val in enumerate(output):
    #     for j, y in enumerate(x[1:]):
    #         if y:
    #             print(x[i][j])
    #             x[i][j] = str(round(float(x[i][j]),3))
                
    # print(f"AFTER: {output}")
    # output = map([for x in output[1:] if x)
    
    with open(csv_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        wr.writerows(output)
    
    return output

def export_csv_raw(csv_file, series, metric):
    queries_set = set()
    for s, queries in series.items():
        for query in queries:
            queries_set.add(query)

    header = (
        "query",
        *(s for s in series)
    )
    output = []

    for query in sorted(queries_set):
        out = []
        for s, queries in series.items():
            if query not in queries:
                v = None 
            else:
                v = queries[query].get(metric)

            out.append(v)

        output.append((query.split('/')[-1], *out))
    
    with open(csv_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        wr.writerows(output)
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some stats")
    parser.add_argument("result_folders", nargs='+', type=str, help="Folders where results are stored. Each folder should contain one or more result file. Each different folder will be treated as a different experiment.")
    parser.add_argument("-d", "--sql_folder", default=None, help="Path to the sql queries.")
    parser.add_argument("-m", "--metric", default="cost", choices=["postgres_cost", "gpuqo_cost"], help="Choose what to show.")
    parser.add_argument("-s", "--save", default=None, help="Path to save plot to.")
    parser.add_argument("--csv", default=None, help="Path to save csv to.")
    parser.add_argument("--raw", default=False, action="store_true", help="Output raw data to csv.")
    parser.add_argument("-t", "--type", default="scatter", choices=["scatter", "line", "scatter_line", "hist"], help="Choose plot type: scatter or bar.")
    parser.add_argument("-r", "--ratio", default=False, action="store_true", help="Plot ratio related to the first series.")
    parser.add_argument("--min_tables", type=int, default=0, help="Limit the number of tables to only the ones that have more than this number of tables.")
    parser.add_argument("--max_tables", type=int, default=1e10, help="Limit the number of tables to only the ones that have less than this number of tables.")
    parser.add_argument("--name_depth", type=int, default=1, help="Set depth of series name wrt to folder path (default=1).")
    parser.add_argument("--shift", type=float, default=0.25, help="Set spread of points in scatter and line plots (default 0.25).")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print debug messages.")
    parser.add_argument("--show", type=str, default=None, help="Which series to plot hist for..")
    parser.add_argument("-b", "--baseline", type=int, default=None, help="Set the baseline series for ratio plot.")

    args = parser.parse_args()

    if args.verbose:
        print(args)

    series = {
        folder2series(folder, args.name_depth):
        {
            query:content
            for query, content in load_results_complete(folder, args.sql_folder, args.metric).items()
            if (content['tables'] >= args.min_tables 
                and content['tables'] <= args.max_tables)
        }
        for folder in args.result_folders
    }

    if args.ratio:
        queries_min_cost = defaultdict(lambda:inf)

        if args.baseline is None:
            for s, queries in series.items():
                for query, metrics in queries.items():
                    if metrics.get(args.metric) is None:
                        continue
                    
                    queries_min_cost[query] = min(queries_min_cost[query],
                                                    metrics[args.metric])
        else:
            baseline = folder2series(args.result_folders[args.baseline], 
                                    args.name_depth)
            
            for query, metrics in series[baseline].items():
                if metrics.get(args.metric) is None:
                    continue
                
                queries_min_cost[query] = metrics[args.metric]
            

        for s, queries in series.items():
            for query, metrics in queries.items():
                metrics[f"{args.metric}_ratio"] = ratio(metrics[args.metric],
                                                    queries_min_cost[query])
        
        metric = args.metric + "_ratio"

        if args.verbose:
            skip = max(len(s) for s in queries_min_cost)
            print(" "*(skip+2), end='')
            for s, queries in series.items():
                label = s.replace("gpuqo_", "").replace("_postgres_","")
                print(f"{label[:min(len(label), 8)]:8s}", end='')
                print("  ", end='')
            print('\n', end='')
            
            for query, min_cost in queries_min_cost.items():
                print(f"%{skip}s" % query, end='')
                print("  ", end='')
                for s, queries in series.items():
                    if query in queries:
                        v = queries[query].get(metric)
                    else:
                        v = None
                    if v is None:
                        print(" "*8, end='')
                    else:
                        print(f"{v:8.2f}", end='')
                    print('  ', end='')
                print('\n', end='')
    else:
        metric = args.metric

    if args.verbose:
        print(series)

    # if args.type == 'scatter':
    #     scatter_plot(
    #         series, 
    #         metric=metric, 
    #         max_shift=args.shift
    #     )
    # elif args.type == 'line':
    #     line_plot(
    #         series, 
    #         metric=metric, 
    #         max_shift=args.shift
    #     )
    # elif args.type == 'scatter_line':
    #     scatter_line_plot(
    #         series, 
    #         metric=metric, 
    #         max_shift=args.shift
    #     )
    # elif args.type == 'hist':
    #     if not args.show:
    #         print("--show is required")
    #         exit(1)
    #     plt.hist([i[metric] for i in series[args.show].values()])

    if args.csv:
        if args.raw:
            export_csv_raw(args.csv, series, metric)
        else:
            export_csv_aggr(args.csv, series, metric)

    # if args.save:
    #     plt.savefig(args.save)
    # else:
    #     plt.show()
