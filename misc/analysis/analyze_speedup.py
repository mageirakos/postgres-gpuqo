import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats.mstats import gmean
from itertools import cycle

from analyze import load_results, folder2series, format_func

def speedup(v, ref):
    return ref/v

def folder2series_nthreads(folder_name, depth=1):
    name = folder2series(folder_name, depth)
    splits = name.split('_')
    for i, s in enumerate(splits):
        if s[-1] == 't' and s[:-1].isdigit():
            series_name = '_'.join(splits[:i]+splits[i+1:])
            return series_name, int(s[:-1])

    return name, 1

def plot(series, metric="plan", speedup_baseline=None, minor_ticks=[]):
    markers = cycle(['+','x','1','2','3','4',])
    for i, (s, queries) in enumerate(series.items()):
        color=f"C{i+1}"
        marker=next(markers)
        x = sorted(list(queries.keys()))
        x_line = []
        y_line = []
        y_err = []

        y = np.array([queries[k][f'{metric}_time_speedup'] for k in x])

        for n in np.sort(np.unique(x)):
            mask = (x == n) & (y > 0)
            if np.any(mask):
                x_line.append(n)
                y_line.append(gmean(y[mask]))   
                y_err.append([np.min(y[mask]), np.max(y[mask])]) 

        plt.plot(x_line, y_line, label=s, color=color, marker=marker)

    # configure matplotlib
    plt.ylabel("Time speedup (higher is better)")
    plt.yscale("log")
    vals = np.array(minor_ticks)
    ticks = []
    ticks += (1/vals[::-1]).tolist()
    ticks += vals.tolist()
    plt.gca().yaxis.set_minor_locator(plt.FixedLocator(ticks))
    plt.gca().yaxis.set_minor_formatter(plt.FuncFormatter(format_func))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.axhline(1, label=speedup_baseline)

    plt.legend()
    plt.grid(which="major")
    plt.grid(which="minor", linestyle="--")
    plt.xlabel("Number of threads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some stats")
    parser.add_argument("result_folders", nargs='+', type=str, help="Folders where results are stored. Each folder should contain one or more result file. Each different folder will be treated as a different experiment.")
    parser.add_argument("-q", "--query", type=str, help="Query name.")
    parser.add_argument("-s", "--save", default=None, help="Path to save plot to.")
    parser.add_argument("-m", "--metric", default="plan", choices=["plan", "exec", "total", "gpuqo"], help="Show whether to choose plan or execution time or their sum.")
    parser.add_argument("-b", "--baseline", type=int, default=0, help="Set the baseline series for speedup.")
    parser.add_argument("-B", "--baseline_self", default=False, action="store_true", help="Set the baseline series for speedup.")
    parser.add_argument("--name_depth", type=int, default=1, help="Set depth of series name wrt to folder path (default=1).")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print debug messages.")
    parser.add_argument("--ticks", default=[1.5,2,3,4,5,8,15,25,50,75,150,200,250,300], type=float, nargs="+", help="Minor ticks for speedup plot.")


    args = parser.parse_args()

    if args.verbose:
        print(args)

    series = defaultdict(dict)

    for folder in args.result_folders:
        name, nthreads = folder2series_nthreads(folder, args.name_depth)
        results = load_results(folder)
        for query, content in results.items():
            if args.query in query:
                series[name][nthreads] = content
                break

    speedup_baseline = None
    if not args.baseline_self:
        baseline, _ = folder2series_nthreads(args.result_folders[args.baseline]
        , args.name_depth)
    else:
        baseline = "self"
    new_series = {}

    for s, results in list(series.items()):
        new_label = s
        new_series[new_label] = {}
        
        if args.baseline_self:
            baseline = s
        
        if len(results) == 1:
            continue

        for nthread in results:
            new_series[new_label][nthread] = {
                f"{metric}_time_speedup": speedup(
                    results[nthread][f"{metric}_time_avg"], 
                    series[baseline][1][f"{metric}_time_avg"]
                )
                for metric in ['plan', 'exec', 'total', 'gpuqo']
            }
    
    if args.baseline_self:
        baseline = "self"

    series = new_series
    speedup_baseline = baseline

    if args.verbose:
        print(series)

    plot(series, args.metric, baseline, args.ticks)    

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
    