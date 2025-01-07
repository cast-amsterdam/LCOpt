import sys
import time
from modified_pygad import pygad
#import pygad
from matplotlib import pyplot as plt

from interface import interface_pygad
from plot_result_optimization import plot_result

import globals

def ga(crf_name, sample, wet, iters, segments):
    """
    Run the genetic algorithm for a given number of
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

    num_parents_mating = 4
    sol_per_pop = globals.popsize
    parent_selection_type = "sss"
    mutation_type = "random"
    mutation_probability = 0.4
    iters = iters-1
    bounds = []
    # Add phi bounds
    for i in range(segments + 1):
        bounds.append({'low': 0.0, 'high': 1.0})
    # Add t_init bounds
    bounds.append({'low': 0.0, 'high': globals.t_init_bound})

    # add delta_t bounds
    for i in range(segments):
        bounds.append({'low': 0.1, 'high': globals.delta_t_bound})

    num_genes = len(bounds)

    # Define your objective function with extra values
    def objective_function_wrapper(crf_name=crf_name, wet=wet, sample_name=sample):
        def objective_function(solution, solution_index):
            result = interface_pygad(crf_name, sample_name, wet, solution, segments, 'genalgo', solution_index)
            return result
        return objective_function


    ga_instance = pygad.GA(num_generations=iters,
                           num_parents_mating=num_parents_mating,
                           fitness_func=objective_function_wrapper(crf_name=crf_name, wet=wet, sample_name=sample),
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           gene_space=bounds,
                           keep_parents=2,
                           save_best_solutions=True)

    # Record starting time
    start_time = time.time()

    # Run the differential evolution algorithm
    ga_instance.run()

    runtime = time.time() - start_time

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    all_solutions = ga_instance.best_solutions
    runtime_per_iteration = ga_instance.runtimes
    return_list = [iters, runtime, solution_fitness, solution, ga_instance.best_solutions_fitness, runtime_per_iteration, all_solutions]
    print(len(ga_instance.best_solutions_fitness))
    return return_list


def main():
    if len(sys.argv) > 6:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 6:
        print('Please specify the crf name, sample name, wet/dry, number of iterations and number of segments, e.g:'
              'python ga.py sum_of_res sample1 True 10 2')
        sys.exit()

    crf_name = sys.argv[1]
    sample = sys.argv[2]
    wet = sys.argv[3]
    iters = int(sys.argv[4])
    segments = int(sys.argv[5])

    result_list = ga(crf_name, sample, wet, iters, segments)

    print("Best CRF score found: ", result_list[2])
    print("Best solution found: ", result_list[3])
    print("Total runtime: ", result_list[1], " seconds")
    print("Number of generations: ", result_list[0])
    print("Number of segments in the gradient profile: ", segments)
    fig, ax, ax2 = plot_result(result_list, sample)
    # show the plot
    plt.show()


if __name__ == '__main__':
    main()
