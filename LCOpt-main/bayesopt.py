import sys
import time

from matplotlib import pyplot as plt

import interface
from modified_skopt.gp import gp_minimize
from interface import interface
from plot_result_optimization import plot_result
from functools import partial

import globals

def bayesopt(crf_name, sample, wet, iters, segments):
    """
    Run the Bayesian optimization algorithm for a given number of
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
             cumulative runtime after each generation. all_solutions is a list of all
                solutions found during the run.

    """
    # We need to create the list of bounds for every parameter.
    bounds = []
    # Add phi bounds
    for i in range(segments + 1):
        bounds.append((0.0, 1.0))
    # Add t_init bounds
    bounds.append((0.0, globals.t_init_bound))
    # Add delta_t bounds
    for i in range(segments):
        bounds.append((0.1, globals.delta_t_bound))

    objective = partial(interface, crf_name=crf_name, wet=wet, sample_name=sample, segments=segments, algorithm="bayesopt")

    # Record starting time
    start_time = time.time()

    # Run Bayesian optimization
    res, runtimes_cumulative = gp_minimize(
                      objective,                # the function to minimize
                      bounds,                   # the bounds on each dimension of x
                      acq_func="EI",            # the acquisition function
                      n_calls=iters + globals.popsize,       # the number of evaluations of f
                      n_initial_points=globals.popsize)      # the number of random initialization points

    # Record ending time
    runtime = time.time() - start_time

    func_vals = [-1 * score for score in res.func_vals]
    return_list = [iters, runtime, -res.fun, list(res.x), func_vals, runtimes_cumulative, res.x_iters]
    return(return_list)


def main():
    if len(sys.argv) > 6:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 6:
        print('Please specify the crf name, sample name, wet/dry, number of iterations and number of segments, e.g.:'
              'python bayesopt.py sum_of_res sample1 True 10 2')
        sys.exit()

    crf_name = sys.argv[1]
    sample = sys.argv[2]
    wet = sys.argv[3]
    iters = int(sys.argv[4])
    segments = int(sys.argv[5])

    result_list = bayesopt(crf_name, sample, wet, iters, segments)

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
