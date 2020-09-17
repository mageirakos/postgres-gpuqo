#!/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T
from scipy.stats.mstats import gmean
from itertools import cycle

def file2series(file):
    return file.split('/')[-2]

def load_results(filename):
    queries = {}
    with open(filename, 'r') as f:
        query = None
        iteration = 0
        iteration_time = 0
        steps_time = {}
        
        for line in f.readlines():
            try:
                if 'sql' in line:
                    query = line[:-5]
                    if query not in queries:
                        queries[query] = {
                            'iterations': {}
                        }

                elif 'Starting' in line:
                    iteration = int(line.split(':')[0].split()[-1])
                    steps_time = {}
                elif 'took' in line:
                    if 'iteration' in line:
                        iteration_time = float(line.split()[2][:-2])
                        steps_tot_time = sum(steps_time.values())
                        steps_time['other'] = iteration_time-steps_tot_time

                        queries[query]['iterations'][iteration] = {
                            'time': iteration_time,
                            'steps': steps_time
                        }

                        steps_time = {}
                    elif 'gpuqo' in line:
                        tot_time = float(line.split()[2][:-2])
                        queries[query]['time'] = tot_time
                        steps_tot_time = sum(steps_time.values())
                        steps_time['other'] = tot_time-steps_tot_time
                        queries[query]['steps'] = steps_time

                        steps_time = {}
                    else:
                        step = line.split()[0]
                        step_time = float(line.split()[2][:-2])
                        steps_time[step] = step_time
                else:
                    print(f"Ignoring line: {line}")
            except Exception as e:
                print(e)
                print("Error in line")
                print(line)

    return queries

def load_results_filter(filename, query_filter):    
    queries = load_results(filename)
    return [
        content 
        for query, content in queries.items() 
        if query_filter in query
    ][0]


def bar_plot(series, query):
    keys = sorted(list(next(iter(series.values()))['iterations']))

    color_cycle = cycle(f"C{i}" for i in range(10))
    colors = {}

    n_series = len(series)
    n_bars = len(keys)
    base_x = np.arange(n_bars)  # the label locations
    width = 1/(n_series+1)  # the width of the bars

    fig, ax = plt.subplots()

    begin = - (n_series - 1) * width / 2
    for i, s in enumerate(series.keys()):
        query = series[s]
        xs = base_x + begin + i*width
        y = []
        for x, iteration in zip(xs,query['iterations']):
            bottom = 0
            for step, time in query['iterations'][iteration]['steps'].items():
                label = f"{s}_{step}"
                if step not in colors:
                    colors[step] = next(color_cycle)
                plt.bar(x, time, width, bottom=bottom, label=step, color=colors[step])
                bottom += time
            y.append(query['iterations'][iteration]['time'])
        plt.plot(xs, y, label=s)

    ax.grid()
    ax.set_xlabel("Iteration ID")
    ax.set_ylabel("Time (ms)")

    handles, labels = plt.gca().get_legend_handles_labels()

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some stats")
    parser.add_argument("query", nargs=1, type=str, help="Name of the query to plot.")
    parser.add_argument("result_files", nargs='+', type=str, help="Files with profiling results.")
    parser.add_argument("-s", "--save", default=None, help="Path to save plot to.")

    args = parser.parse_args()

    series = {
        file2series(filename):
        load_results_filter(filename, args.query[0])
        for filename in args.result_files
    }

    bar_plot(series, args.query)

    min_iters = {}
    for s, query in series.items():
        print("%s %.2f(%.2f)" % (
            s,
            query['time'], 
            sum(query['iterations'][i]['time'] for i in query['iterations'])
        ))

        for step, time in query['steps'].items():
            print(f"    {step:16s} {time:.2f}ms\t{time/query['time']*100:5.2f}%")

        for i in query['iterations']:
            if i in min_iters:
                min_iters[i] = min(min_iters[i], query['iterations'][i]['time'])
            else:
                min_iters[i] = query['iterations'][i]['time']

    print("min %.2f" % (
        sum(min_iters.values())
    ))

    if args.save:
        plt.save(args.save)
    else:
        plt.show()
    