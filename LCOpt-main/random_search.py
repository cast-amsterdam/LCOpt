import sys
import math
import time
import random

from matplotlib import pyplot as plt

import interface
import numpy as np
from plot_result_optimization import plot_result

import globals

def run_rs(crf_name, sample, wet, iters, segments):
    """
    Run the random_search algorithm for a given number of
    iterations and gradient profile segments.

    :crf_name: Name of the CRF to use.
    :sample: Name of the sample to use.
    :wet: Whether to use the "dry" or "wet" setting.
    :iters: Number of iterations the algorithm should perform.
    :segments: Number of gradient segments in the gradient profile.
    :return: A list of the form
             [iters, runtime, solution_fitness, solution, crf_per_iteration, runtime_per_iteration, all_solutions]
             where iters is the number of iterations, runtime is the total runtime
             for all iterations, solution_fitness is the best crf score found, solution is
             the best solution found, crf_per_iteration is the best crf score found
             so far after every generation and runtime_per_iteration is the
             cumulative runtime after each generation. all_solutions is a list of
                all solutions found.

    """

    bounds = []

    # Add phi bounds
    for i in range(segments + 1):
        bounds.append([0.0, 1.0])
    # Add t_init bounds
    bounds.append([0.0, globals.t_init_bound])
    for i in range(segments):
        bounds.append([0.1, globals.delta_t_bound])

    best_performance = math.inf

    runtime_per_iteration = []
    # Record starting time
    start_time = time.time()
    best_score_per_iteration_list = []
    best_solution_per_iteration_list = []
    # store all solutions generated
    all_solutions = []
    for i in range(iters + globals.popsize):
        # Get random chromosome
        chromosome = []
        for gene_range in bounds:
            lower_bound = gene_range[0]
            upper_bound = gene_range[1]
            chromosome.append(random.uniform(lower_bound, upper_bound))

        all_solutions.append(chromosome)
        chromosome = np.array(chromosome)

        # Run the objective function for that set of parameters
        current_performance = interface.interface(chromosome, crf_name=crf_name, wet=wet,
                                                  sample_name=sample, segments=segments, algorithm='rs')
        # If the performance is better than the best performance so far, (minimize)
        if(current_performance < best_performance):
            # Keep the parameters and performance
            best_performance = current_performance
            best_parameters = chromosome
        best_score_per_iteration_list.append(-best_performance)
        best_solution_per_iteration_list.append(best_parameters)
        runtime = time.time() - start_time
        runtime_per_iteration.append(runtime)

    return_list = [iters, runtime, -best_performance, best_parameters, best_score_per_iteration_list, runtime_per_iteration, all_solutions]
    return(return_list)


def main():
    if len(sys.argv) > 6:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 6:
        print('Please specify the crf name, sample name, wet/dry, number of iterations and number of segments, e.g.:'
              'python random_search.py sum_of_res sample_real True 10 2')
        sys.exit()

    crf_name = sys.argv[1]
    sample = sys.argv[2]
    wet = sys.argv[3]
    iters = int(sys.argv[4])
    segments = int(sys.argv[5])

    result_list = run_rs(crf_name, sample, wet, iters, segments)

    print("Best CRF score found: ", result_list[2])
    print("Best solution found: ", result_list[3])
    print("Total runtime: ", result_list[1], " seconds")
    print("Number of iterations: ", result_list[0])
    print("Number of segments in the gradient profile: ", segments)
    fig, ax, ax2 = plot_result(result_list, sample)
    # show the plot
    plt.show()


if __name__ == '__main__':
    main()
