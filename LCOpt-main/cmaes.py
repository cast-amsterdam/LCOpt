import sys

import cma
import time

from matplotlib import pyplot as plt

from interface import interface
from plot_result_optimization import plot_result

import globals

def cmaes(crf_name, sample, wet, iters, segments):
    """
    Run the CMA-ES algorithm for a given number of
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
    pop_size = 10
    iters = iters+1
    lower_bounds = []
    upper_bounds = []
    # Add phi bounds
    for i in range(segments + 1):
        lower_bounds.append(0.0)
        upper_bounds.append(1.0)
    # Add t_init bounds
    lower_bounds.append(0.0)
    upper_bounds.append(globals.t_init_bound)
    # Add delta_t bounds
    for i in range(segments):
        lower_bounds.append(0.1)
        upper_bounds.append(globals.delta_t_bound)

    num_pars = len(lower_bounds)
    es = cma.CMAEvolutionStrategy(num_pars * [0.1], 0.5, {'popsize':globals.popsize, 'maxiter':iters, 'bounds':[lower_bounds,upper_bounds]})

    # Record starting time
    start_time = time.time()
    runtimes = []
    func_vals = []
    all_solutions = []
    # note: assumes minimization
    for i in range(iters):
        solutions = es.ask()
        f = [interface(x, crf_name, wet, sample, segments, 'cmaes') for x in solutions]
        func_vals.append(f)
        es.tell(solutions, f)
        runtimes.append(es.timer.elapsed)
        all_solutions.append(solutions)

    # flatten
    func_vals = [item for row in func_vals for item in row]
    all_solutions = [item for row in all_solutions for item in row]
    # Record ending time
    runtime = time.time() - start_time
    return_list = [iters, runtime, -es.best.f,  es.best.x, [-1 * f for f in func_vals], runtimes, all_solutions]
    return return_list


def main():
    if len(sys.argv) > 6:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 6:
        print('Please specify the crf name, sample name, wet/dry, number of iterations and number of segments, e.g.:'
            'python cmaes.py sum_of_res sample1 True 10 2')
        sys.exit()

    crf_name = sys.argv[1]
    sample = sys.argv[2]
    wet = sys.argv[3]
    iters = int(sys.argv[4])
    segments = int(sys.argv[5])

    result_list = cmaes(crf_name, sample, wet, iters, segments)

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
