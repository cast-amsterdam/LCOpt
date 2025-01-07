import sys
import time
from matplotlib import pyplot as plt
from scipy import optimize
from interface import interface
from plot_result_optimization import plot_result
from functools import partial

import globals

def grid_search(crf_name, sample, wet, Ns, segments):
    """
    Run the grid search algorithm for a given number of
    grid points in each dimension and gradient profile segments.

    :crf_name: Name of the CRF to use.
    :sample: Name of the sample to use.
    :wet: Whether to use the "dry" or "wet" setting.
    :Ns: Number of gradient points in each dimension.
    :segments: Number of gradient segments in the gradient profile.

    Total number of grid points is (Ns ** (2*segments + 2))

    :return: A list of the form
             [iters, runtime, solution_fitness, solution, params_array, values_array]
             where iters is the number of iterations, runtime is the total runtime
             for all iterations, solution_fitness is the best crf score found, solution is
             the best solution found. params_array is an array of all evaluated parameters
                and values_array is an array of all evaluated values.
    """


    rranges = []

    # Add phi bounds
    for i in range(segments + 1):
        rranges.append((0.01, 1.0))
    # Add t_init bounds
    rranges.append((0.01, globals.t_init_bound))
    for i in range(segments):
        rranges.append((0.01, globals.delta_t_bound))

    rranges = tuple(rranges)

    objective = partial(interface, crf_name=crf_name, wet=wet, sample_name=sample, segments=segments, algorithm='gs')

    start_time = time.time()

    resbrute = optimize.brute(objective, rranges, Ns=Ns, full_output=True, finish=None)

    runtime = time.time() - start_time

    iters = len(rranges) ** 5

    # reshape grid and values
    params_array = resbrute[2].reshape(len(rranges), -1).T  # Shape: [n_grid_points, n_pars]
    values_array = resbrute[3].flatten()  # Shape: [n_grid_points]
    result_list = [iters, runtime, -resbrute[1], resbrute[0], params_array, -values_array]
    return(result_list)


def main():
    if len(sys.argv) > 6:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 6:
        print('Please specify the crf name, sample name, wet/dry, number of grid points and number of segments, e.g.:'
              'python grid_search.py 10 2')
        sys.exit()

    crf_name = sys.argv[1]
    sample = sys.argv[2]
    wet = sys.argv[3]
    grid_points_per_dimension = int(sys.argv[4])
    number_of_segments = int(sys.argv[5])

    grid_points_total = grid_points_per_dimension ** (2*number_of_segments + 2)
    print("Total number of grid points to compute: ", grid_points_total)

    result_list = grid_search(crf_name, sample, wet, grid_points_per_dimension, number_of_segments)

    print("Best CRF score found: ", result_list[2])
    print("Best solution found: ", result_list[3])
    print("Total runtime: ", result_list[1], " seconds")
    print("Number of grid points (total): ", grid_points_total)
    print("Number of segments in the gradient profile: ", number_of_segments)
    fig, ax, ax2 = plot_result(result_list, sample)
    # show the plot
    plt.show()

if __name__ == '__main__':
    main()



