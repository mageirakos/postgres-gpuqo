#!/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T

def folder2series(folder, depth=1):
    return '/'.join(folder.split('/')[-depth:])

def ratio(v, ref):
    return v/ref
    if v > ref:
        return v/ref
    elif v < ref:
        return -ref/v
    else:
        return 0


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
            plan_time = 0
            exec_time = 0
            for line in f.readlines():
                if 'sql' in line:
                    if query and (plan_time or exec_time):
                        # save if there was a query previously
                        # create dict item if missing
                        if query not in queries:
                            queries[query] = {
                                'plan_time_raw': [],
                                'exec_time_raw': [],
                                'total_time_raw': [],
                            }

                        # fill dict item
                        queries[query]['plan_time_raw'].append(plan_time)
                        queries[query]['exec_time_raw'].append(exec_time)
                        queries[query]['total_time_raw'].append(plan_time + exec_time)

                        # reset values
                        plan_time = 0
                        exec_time = 0
                    query = line[:-5]
                elif 'Planning' in line:
                    plan_time = float(line.split(':')[1][:-3])
                elif 'Execution' in line:
                    exec_time = float(line.split(':')[1][:-3])

                else:
                    print(f"Unexpected line: {line}")

            if query:
                # save last query if any
                # create dict item if missing
                if query not in queries:
                    queries[query] = {
                        'plan_time_raw': [],
                        'exec_time_raw': [],
                        'total_time_raw': [],
                    }

                # fill dict item
                queries[query]['plan_time_raw'].append(plan_time)
                queries[query]['exec_time_raw'].append(exec_time)
                queries[query]['total_time_raw'].append(plan_time + exec_time)
    
    for query, d in queries.items():
        for metric in ['plan', 'exec', 'total']:
            d[f"{metric}_time_avg"] = np.mean(d[f"{metric}_time_raw"])
            d[f"{metric}_time_std"] = np.std(d[f"{metric}_time_raw"])
            d[f"{metric}_time_count"] = len(d[f"{metric}_time_raw"])

    return queries

def add_table_count(queries, sql_folder):
    for query, d in queries.items():
        d['tables'] = count_tables(
            read_query(
                os.path.join(sql_folder, query + '.sql')
                )
            )
    return queries

def load_results_complete(result_folder, sql_folder):
    queries = load_results(result_folder)
    add_table_count(queries, sql_folder)
    return queries

def scatter_plot_count_time(queries, metric="plan", label=None, ratio=False, color=None):
    keys = sorted(list(queries.keys()))
    x = [queries[k]['tables'] for k in keys]
    if ratio:
        y = [queries[k][f'{metric}_time_ratio'] for k in keys]
    else:
        y = [queries[k][f'{metric}_time_avg'] for k in keys]
        yerr = [queries[k][f'{metric}_time_std'] for k in keys]
        plt.errorbar(x, y, yerr=yerr, linestyle="None", color=color)

    plt.scatter(x, y, label=label, color=color)

def scatter_plot(series, metric="plan", ratio=False):
    for i, (s, queries) in enumerate(series.items()):
        scatter_plot_count_time(
            queries, 
            metric=args.metric, 
            label=s, 
            ratio=ratio,
            color=f"C{i if not ratio else i+1}")

    # configure matplotlib
    if ratio:
        plt.ylabel("Time ratio (lower is better)")
        plt.yscale("log")
        vals = np.array([2,3,5,10,100])
        ticks = []
        labels = []
        ticks += (1/vals[::-1]).tolist()
        labels += [f"1/{i}" for i in vals[::-1]]
        ticks.append(1)
        labels.append("1")
        ticks += vals.tolist()
        labels += [str(i) for i in vals]
        plt.yticks(ticks, labels)
        plt.axhline(1, label=next(iter(series)).split(' vs ')[1])
    else:
        plt.ylabel("Time (ms)")

    plt.legend()
    plt.grid()
    plt.xlabel("Number of tables")

def bar_plot(series, metric="plan", ratio=False):
    keys = sorted(list(next(iter(series.values())).keys()))

    n_series = len(series)
    n_bars = len(keys)
    base_x = np.arange(n_bars)  # the label locations
    width = 1/(n_series+1)  # the width of the bars

    fig, ax = plt.subplots()

    begin = - (n_series - 1) * width / 2
    for i, s in enumerate(series.keys()):
        queries = series[s]
        x = base_x + begin + i*width
        if ratio:
            y = [queries[k][f'{metric}_time_ratio'] for k in keys]
            plt.bar(x, y, width, label=s)
        else:
            y = [queries[k][f'{metric}_time_avg'] for k in keys]
            yerr = [queries[k][f'{metric}_time_std'] for k in keys]
            plt.bar(x, y, width, label=s)
            plt.errorbar(x, y, yerr=yerr, linestyle="None")

    # configure matplotlib
    if ratio:
        ax.axhline(-1, color="red", label="Below this line series is faster than reference")
        ax.axhline(1, color="green", label="Above this line series is slower than reference")
        ax.axhline(0, color="blue", label="Equality line")
    ax.legend()
    ax.grid()
    ax.set_xlabel("Query (#tables)")
    if ratio:
        ax.set_ylabel("Time ratio\n(Y = series > reference ? series/reference : reference/series)")
    else:
        ax.set_ylabel("Time (ms)")

    ax.set_xticks(base_x)
    ax.set_xticklabels([f"{k} ({next(iter(series.values()))[k]['tables']})" for k in keys])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some stats")
    parser.add_argument("result_folders", nargs='+', type=str, help="Folders where results are stored. Each folder should contain one or more result file. Each different folder will be treated as a different experiment.")
    parser.add_argument("-d", "--sql_folder", default=".", help="Path to the sql queries.")
    parser.add_argument("-s", "--save", default=None, help="Path to save plot to.")
    parser.add_argument("-m", "--metric", default="plan", choices=["plan", "exec", "total"], help="Show whether to choose plan or execution time or their sum.")
    parser.add_argument("-t", "--type", default="scatter", choices=["scatter", "bar"], help="Choose plot type: scatter or bar.")
    parser.add_argument("-r", "--ratio", default=False, action="store_true", help="Plot ratio related to the first series.")
    parser.add_argument("--min_tables", type=int, default=0, help="Limit the number of tables to only the ones that have more than this number of tables.")
    parser.add_argument("--max_tables", type=int, default=1e10, help="Limit the number of tables to only the ones that have less than this number of tables.")
    parser.add_argument("--name_depth", type=int, default=1, help="Set depth of series name wrt to folder path (default=1).")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print debug messages.")

    args = parser.parse_args()

    if args.verbose:
        print(args)

    series = {
        folder2series(folder, args.name_depth):
        {
            query:content
            for query, content in load_results_complete(folder, args.sql_folder).items()
            if (content['tables'] >= args.min_tables 
                and content['tables'] <= args.max_tables)
        }
        for folder in args.result_folders
    }

    if args.ratio:
        baseline = folder2series(args.result_folders[0], args.name_depth)
        new_series = {}
        for s, queries in series.items():
            if s == baseline:
                continue
            new_label = f"{s} vs {baseline}"
            new_series[new_label] = {}
            for query in queries:
                new_series[new_label][query] = {
                    f"{metric}_time_ratio": ratio(
                        queries[query][f"{metric}_time_avg"], 
                        series[baseline][query][f"{metric}_time_avg"]
                    ) if query in series[baseline] else np.nan
                    for metric in ['plan', 'exec', 'total']
                }
                new_series[new_label][query]['tables'] =queries[query]['tables']

        series = new_series

    if args.verbose:
        print(series)

    if args.type == 'scatter':
        scatter_plot(
            series, 
            metric=args.metric, 
            ratio=args.ratio
        )
    elif args.type == 'bar':
        bar_plot(
            series, 
            metric=args.metric, 
            ratio=args.ratio
        )

    if args.save:
        plt.save(args.save)
    else:
        plt.show()
    
    