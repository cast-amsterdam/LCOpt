import os
import csv
import sys
import time
import json

import numpy as np

import diffevo
import bayesopt
import ga
import interface
import random_search
import grid_search
import cmaes
from tqdm import tqdm
import matplotlib.pyplot as plt

import globals

from plot_result_optimization import plot_result

def run_n_times(algorithm, crf_name, sample, wet, segments, n, iters):
    """
    Perform a meta-experiment. The chosen algorithm is run n times for a given
    number of gradient segments. The CRF score per iteration and the cumulative
    runtime per iteration are written to csv files. Filepaths need to be specified
    manually and should indicate which sample was used in read_data.py. This has
    to be done manually because optimization algorithm packages don't allow
    for extra arguments in the objective function, other than the parameters
    to be optimized.

    :algorithm: Optimization algorithm. Choose from:
                BayesOpt/DiffEvo/GenAlgo/GridSearch/RandomSearch
    :segments: Number of gradient segments in the gradient profile.
    """

    if(wet == "True"):
        prefix = globals.results_folder + "/wet/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"
    else:
        prefix = globals.results_folder + "/dry/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"

    filename_score = "score" + ".csv"
    filename_runtime = "runtime" + ".csv"
    filename_solution = "solution" + ".csv"
    filename_purity = "purity" + ".csv"

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    filepath_score = prefix + filename_score
    filepath_runtime = prefix + filename_runtime
    filepath_solution = prefix + filename_solution
    filepath_purity = prefix + filename_purity

    figname = "best_solution_chrom" + ".png"
    filepath_fig = prefix + figname
    if(algorithm == "BayesOpt"):
        for nth_experiment in tqdm(range(n)):
            interface.set_counter(0)
            counter = interface.get_counter()
            figname = "best_solution_chrom_" + str(nth_experiment) + ".png"
            filepath_fig = prefix + figname

            solname = "all_solutions_" + str(nth_experiment) + ".csv"
            filepath_sol = prefix + solname

            # n = number of meta experiments
            return_list = bayesopt.bayesopt(crf_name, sample, wet, iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]
            # write the data from the list to a (csv?) file as a single line

            f = open(filepath_score, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)

            # write all solutions to file.
            f = open(filepath_sol, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerows(return_list[6])

            peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'bayesopt')
            print("Peak purity: ", -peak_purity)
            min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'bayesopt')
            print("Min time: ", min_time)
            print("Max time: ", max_time)

            # write peak purity and min and max time to same file
            f = open(filepath_purity, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow([-peak_purity, min_time, max_time])

            fig, ax, ax2= plot_result(return_list, sample)
            # use filepath_fig to save the plot
            plt.savefig(filepath_fig)


    elif(algorithm == "DiffEvo"):
        for nth_experiment in tqdm(range(n)):
            interface.set_counter(0)
            figname = "best_solution_chrom_" + str(nth_experiment) + ".png"
            filepath_fig = prefix + figname

            solname = "all_solutions_" + str(nth_experiment) + ".csv"
            filepath_sol = prefix + solname

            # n = number of meta experiments
            return_list = diffevo.diffevo(crf_name, sample, wet, iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]
            all_solutions = return_list[6]

            # DiffEvo does not return the lowest value of the initial population, but we do need to store it.
            # load scores_init.txt
            scores_init = np.loadtxt(prefix + "scores_init.txt")[nth_experiment*globals.popsize:nth_experiment*globals.popsize+globals.popsize]
            min_scores_init = np.min(scores_init)
            # get index of min_scores_init in func_vals
            index_min_scores_init = np.argmin(scores_init)

            # place min_scores_init at the beginning of func_vals
            func_vals = np.insert(func_vals, 0, min_scores_init)

            f = open(filepath_score, 'a', newline ='\n')

            # Diffevo does not return the solution of the lowest value of the initial population, but we do need to store it.
            # load solutions_init.txt
            solutions_init = np.loadtxt(prefix + "profiles_init.txt", delimiter=',')[nth_experiment*globals.popsize:nth_experiment*globals.popsize+globals.popsize]
            min_solution_init = solutions_init[index_min_scores_init]

            # place solutions_init at the beginning of all_solutions
            all_solutions = np.insert(all_solutions, 0, min_solution_init, axis=0)

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)

            peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'diffevo')
            print("Peak purity: ", -peak_purity)
            min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'diffevo')
            print("Min time: ", min_time)
            print("Max time: ", max_time)

            # write peak purity and min and max time to same file
            f = open(filepath_purity, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow([-peak_purity, min_time, max_time])

            fig, ax, ax2= plot_result(return_list, sample)
            # use filepath_fig to save the plot
            plt.savefig(filepath_fig)

            # write all solutions to file.
            f = open(filepath_sol, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerows(all_solutions.tolist())



    elif(algorithm == "GenAlgo"):
        for nth_experiment in tqdm(range(n)):
            interface.set_counter(0)
            figname = "best_solution_chrom_" + str(nth_experiment) + ".png"
            filepath_fig = prefix + figname
            solname = "all_solutions_" + str(nth_experiment) + ".csv"
            filepath_sol = prefix + solname

            return_list = ga.ga(crf_name, sample, wet, iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            # GenAlgo does not return the lowest value of the initial population, but we do need to store it.
            # load scores_init.txt
            scores_init = np.loadtxt(prefix + "scores_init.txt")[
                          nth_experiment * globals.popsize:nth_experiment * globals.popsize + globals.popsize]
            min_scores_init = np.min(scores_init)
            index_min_scores_init = np.argmin(scores_init)

            # place min_scores_init at the beginning of func_vals
            func_vals = np.insert(func_vals, 0, min_scores_init)

            # GenAlgo does not return the solution of the lowest value of the initial population, but we do need to store it.
            # load solutions_init.txt
            solutions_init = np.loadtxt(prefix + "profiles_init.txt", delimiter=',')[
                                nth_experiment * globals.popsize:nth_experiment * globals.popsize + globals.popsize]
            min_solution_init = solutions_init[index_min_scores_init]

            # place solutions_init at the beginning of all_solutions
            all_solutions = np.insert(return_list[6], 0, min_solution_init, axis=0)

            f = open(filepath_score, 'a', newline ='\n')


            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)

            # write all solutions to file.
            f = open(filepath_sol, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerows(all_solutions.tolist())

            peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'genalgo')
            print("Peak purity: ", -peak_purity)
            min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'genalgo')
            print("Min time: ", min_time)
            print("Max time: ", max_time)

            # write peak purity and min and max time to same file
            f = open(filepath_purity, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow([-peak_purity, min_time, max_time])

            fig, ax, ax2 = plot_result(return_list, sample)
            # use filepath_fig to save the plot
            plt.savefig(filepath_fig)

    elif(algorithm == "CMA"):
        for nth_experiment in tqdm(range(n)):
            interface.set_counter(0)
            figname = "best_solution_chrom_" + str(nth_experiment) + ".png"
            filepath_fig = prefix + figname
            solname = "all_solutions_" + str(nth_experiment) + ".csv"
            filepath_sol = prefix + solname

            return_list = cmaes.cmaes(crf_name, sample, wet, iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            f = open(filepath_score, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)

            # write all solutions to file.
            f = open(filepath_sol, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerows(return_list[6])

            peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'cmaes')
            print("Peak purity: ", -peak_purity)
            min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'cmaes')
            print("Min time: ", min_time)
            print("Max time: ", max_time)

            # write peak purity and min and max time to same file
            f = open(filepath_purity, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow([-peak_purity, min_time, max_time])

            fig, ax, ax2 = plot_result(return_list, sample)
            # use filepath_fig to save the plot
            plt.savefig(filepath_fig)

    elif(algorithm == "RandomSearch"):
        for nth_experiment in tqdm(range(n)):
            interface.set_counter(0)
            counter = interface.get_counter()
            figname = "best_solution_chrom_" + str(nth_experiment) + ".png"
            filepath_fig = prefix + figname
            solname = "all_solutions_" + str(nth_experiment) + ".csv"
            filepath_sol = prefix + solname
            # n = number of meta experiments
            return_list = random_search.run_rs(crf_name, sample, wet, iters, segments)
            best_solution = return_list[3]
            func_vals = return_list[4]
            runtime_per_iteration = return_list[5]

            f = open(filepath_score, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(func_vals)

            f = open(filepath_runtime, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow(runtime_per_iteration)

            f = open(filepath_solution, 'a', newline ='\n')

            # writing the data into the file
            with f:
                writer = csv.writer(f)
                writer.writerow(best_solution)

            # write all solutions to file.
            f = open(filepath_sol, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerows(return_list[6])


            peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'randomsearch')
            print("Peak purity: ", -peak_purity)
            min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'randomsearch')
            print("Min time: ", min_time)
            print("Max time: ", max_time)

            # write peak purity and min and max time to same file
            f = open(filepath_purity, 'a', newline ='\n')

            with f:
                writer = csv.writer(f)
                writer.writerow([-peak_purity, min_time, max_time])

            fig, ax, ax2= plot_result(return_list, sample)
            # use filepath_fig to save the plot
            plt.savefig(filepath_fig)

        # Show the plot (optional)
        #plt.show()



    elif(algorithm == "GridSearch"):
        interface.set_counter(0)
        counter = interface.get_counter()
        return_list = grid_search.grid_search(crf_name, sample, wet, iters, segments)
        func_val = return_list[2]
        runtime = return_list[1]
        best_solution = return_list[3]
        solname = "all_solutions_0.csv"
        filepath_sol = prefix + solname
        scorename = "all_scores.csv"
        filepath_score_all = prefix + scorename

        f = open(filepath_score, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow([func_val])

        f = open(filepath_runtime, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow([runtime])

        f = open(filepath_solution, 'a', newline ='\n')

        # writing the data into the file
        with f:
            writer = csv.writer(f)
            writer.writerow(best_solution)

        # write all solutions to file.
        f = open(filepath_sol, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerows(return_list[4])

        # write all scores to file.
        f = open(filepath_score_all, 'a', newline ='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow(return_list[5])



        peak_purity = interface.interface(best_solution, "peak_purity", False, sample, segments, 'gridsearch')
        print("Peak purity: ", -peak_purity)
        min_time, max_time = interface.interface(best_solution, "time_info", False, sample, segments, 'gridsearch')
        print("Min time: ", min_time)
        print("Max time: ", max_time)

        # write peak purity and min and max time to same file
        f = open(filepath_purity, 'a', newline='\n')

        with f:
            writer = csv.writer(f)
            writer.writerow([-peak_purity, min_time, max_time])

        fig, ax, ax2 = plot_result(return_list, sample)
        # use filepath_fig to save the plot
        plt.savefig(filepath_fig)



def main():

    if len(sys.argv) > 8:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 8:
        print('Please specify the following parameters in order:')
        print("- Choose an optimization algorithm (BayesOpt/DiffEvo/GenAlgo/CMA/GridSearch/RandomSearch)")
        print("- Number of segments in the gradient profile")
        print("- Number of sub-experiments the meta-experiment should consist of")
        print("- Number of iterations. Note that if the chosen algorithm is grid search, this is the number of grid points per dimension. For GA and Diffevo this is Func Eval *10, for BO this is Func Eval")
        # For GenAlgo the func evals is func eval *10 +10 +10 for some reason.
        print("- Name of the sample.")
        print("- Wet? (True/False)")
        print("- CRF name (prod_of_res/tyteca11/sum_of_res/tyteca24)")
        sys.exit()

    algorithm = sys.argv[1]
    number_of_segments = int(sys.argv[2])
    sub_experiments = int(sys.argv[3])
    iterations = int(sys.argv[4])
    sample_name = sys.argv[5]
    wet = sys.argv[6]
    crf_name = sys.argv[7]

    # Write variables to json file
    variable_dict = {
        "wet": wet,
        "crf_name": crf_name,
        "sample_name": sample_name,
        "algorithm": algorithm
    }

    # json_object = json.dumps(variable_dict, indent=4)

    # with open("globals_" + algorithm +".json", "w") as outfile:
    #     outfile.write(json_object)

    run_n_times(algorithm, crf_name, sample_name, wet, number_of_segments, sub_experiments, iterations)
    print("RAN FOLLOWING EXPERIMENT:")
    print(algorithm, crf_name, sample_name, wet, number_of_segments, sub_experiments, iterations)
    print("TERMINATION SUCCESFUL")

if __name__ == '__main__':
    main()
