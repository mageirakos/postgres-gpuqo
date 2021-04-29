#!/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T
from scipy.stats.mstats import gmean
from itertools import cycle

def folder2series(folder, depth=1):
    return '/'.join(folder.split('/')[-depth:])

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
            cost = None
            for line in f.readlines():
                if 'sql' in line:
                    if query:
                        if cost is not None:
                            # save if there was a query previously
                            # create dict item if missing
                            if query not in queries:
                                queries[query] = {
                                    'costs': []
                                }

                            # fill dict item
                            queries[query]['costs'].append(cost)

                        # reset values
                        cost = None
                    query = line[:-5]
                else:
                    try:
                        cost = float(line.strip())
                    except ValueError:
                        print(f"Unexpected line: {line.strip()}")

            if query and cost is not None:
                # save if there was a query previously
                # create dict item if missing
                if query not in queries:
                    queries[query] = {
                        'costs': []
                    }

                # fill dict item
                queries[query]['costs'].append(cost)

            # reset values
            cost = None

    for query, d in queries.items():
        for metric in ['cost']:
            d[f"{metric}"] = d[f"{metric}s"][0]

    return queries

def add_table_count(queries, sql_folder):
    for query, d in queries.items():
        d['tables'] = count_tables(
            read_query(
                os.path.join(sql_folder, os.path.basename(query + '.sql'))
                if sql_folder else query + '.sql'
                )
            )
    return queries

def load_results_complete(result_folder, sql_folder):
    queries = load_results(result_folder)
    add_table_count(queries, sql_folder)
    return queries

def scatter_plot_count_cost(queries, metric="plan", label=None, ratio=False, color=None, marker="o", shift=0, marker_size=100):
    keys = sorted(list(queries.keys()))
    x = np.array([queries[k]['tables'] for k in keys]) + shift
    if ratio:
        y = [queries[k][f'{metric}_ratio'] for k in keys]
    else:
        y = [queries[k][f'{metric}'] for k in keys]
        plt.errorbar(x, y, linestyle="None", color=color)

    plt.scatter(x, y, label=label, color=color, marker=marker, s=marker_size)

def line_plot_count_cost(queries, metric="plan", label=None, ratio=False, color=None, errorbar=False):
    keys = sorted(list(queries.keys()))
    x = np.array([queries[k]['tables'] for k in keys])
    x_line = []
    y_line = []
    y_err = []

    if ratio:
        y = np.array([queries[k][f'{metric}_ratio'] for k in keys])

        for n in np.sort(np.unique(x)):
            mask = (x == n) & (y > 0)
            if np.any(mask):
                x_line.append(n)
                print(n, y[mask], gmean(y[mask]))
                y_line.append(gmean(y[mask]))   
                y_err.append([np.min(y[mask]), np.max(y[mask])]) 
    else:
        y = np.array([queries[k][f'{metric}'] for k in keys])

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

def generic_plot(series, line=False, scatter=True, metric="plan", ratio_baseline=None, max_shift=0):
    markers = cycle(['+','x','1','2','3','4',])
    shifts = np.linspace(-max_shift, max_shift, len(series))
    for i, (s, queries) in enumerate(series.items()):
        if scatter:
            scatter_plot_count_cost(
                queries, 
                metric=args.metric, 
                label=s if not line else None, 
                ratio=ratio_baseline is not None,
                color=f"C{i if not ratio_baseline else i+1}", 
                marker=next(markers),
                shift=shifts[i],
                marker_size=10 if line else 40
            )
        if line: 
            line_plot_count_cost(
                queries, 
                metric=args.metric, 
                label=s, 
                ratio=ratio_baseline is not None,
                color=f"C{i if not ratio_baseline else i+1}",
                errorbar=(not scatter)
            )

    # configure matplotlib
    if ratio_baseline:
        plt.ylabel("Cost ratio (lower is better)")
        plt.yscale("log")
        plt.axhline(1, label=ratio_baseline)
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

def bar_plot(series, metric="cost", ratio_baseline=None):
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
        if ratio_baseline:
            y = [queries[k][f'{metric}_ratio'] for k in keys]
            plt.bar(x, y, width, label=s)
        else:
            y = [queries[k][f'{metric}'] for k in keys]
            plt.bar(x, y, width, label=s)
            plt.errorbar(x, y, linestyle="None")

    # configure matplotlib
    ax.legend()
    ax.grid()
    ax.set_xlabel("Query (#tables)")
    if ratio_baseline:
        ax.set_ylabel("Cost ratio\n(Y = series > reference ? series/reference : reference/series)")
    else:
        ax.set_ylabel("Cost")

    ax.set_xticks(base_x)
    ax.set_xticklabels([f"{k.split('/')[-1]} ({next(iter(series.values()))[k]['tables']})" for k in keys])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some stats")
    parser.add_argument("result_folders", nargs='+', type=str, help="Folders where results are stored. Each folder should contain one or more result file. Each different folder will be treated as a different experiment.")
    parser.add_argument("-d", "--sql_folder", default=None, help="Path to the sql queries.")
    parser.add_argument("-m", "--metric", default="cost", choices=["cost"], help="Choose what to show.")
    parser.add_argument("-s", "--save", default=None, help="Path to save plot to.")
    parser.add_argument("-t", "--type", default="scatter", choices=["scatter", "bar","line", "scatter_line"], help="Choose plot type: scatter or bar.")
    parser.add_argument("-r", "--ratio", default=False, action="store_true", help="Plot ratio related to the first series.")
    parser.add_argument("--min_tables", type=int, default=0, help="Limit the number of tables to only the ones that have more than this number of tables.")
    parser.add_argument("--max_tables", type=int, default=1e10, help="Limit the number of tables to only the ones that have less than this number of tables.")
    parser.add_argument("--name_depth", type=int, default=1, help="Set depth of series name wrt to folder path (default=1).")
    parser.add_argument("--shift", type=float, default=0.25, help="Set spread of points in scatter and line plots (default 0.25).")
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

    ratio_baseline = None
    if args.ratio:
        baseline = folder2series(args.result_folders[0], args.name_depth)
        new_series = {}
        for s, queries in series.items():
            if s == baseline:
                continue
            new_label = s
            new_series[new_label] = {}
            for query in queries:
                new_series[new_label][query] = {
                    f"{metric}_ratio": ratio(
                        queries[query][f"{metric}"], 
                        series[baseline][query][f"{metric}"]
                    ) if query in series[baseline] else np.nan
                    for metric in ['cost']
                }
                new_series[new_label][query]['tables'] =queries[query]['tables']

        series = new_series
        ratio_baseline = baseline

    if args.verbose:
        print(series)

    if args.type == 'scatter':
        scatter_plot(
            series, 
            metric=args.metric, 
            ratio_baseline=ratio_baseline,
            max_shift=args.shift
        )
    elif args.type == 'bar':
        bar_plot(
            series, 
            metric=args.metric, 
            ratio_baseline=ratio_baseline
        )
    elif args.type == 'line':
        line_plot(
            series, 
            metric=args.metric, 
            ratio_baseline=ratio_baseline,
            max_shift=args.shift
        )
    elif args.type == 'scatter_line':
        scatter_line_plot(
            series, 
            metric=args.metric, 
            ratio_baseline=ratio_baseline,
            max_shift=args.shift
        )

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
    