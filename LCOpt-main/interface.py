import csv

import crf
import retention_model
import read_data as rd
import math
import json
import numpy as np
import peak_and_width_detection as pwd

import globals

# HPLC system parameters
t_0 = globals.t_0
t_D = globals.t_D
N = globals.N

def chromosome_to_lists(chromosome):
    """
    Transform a candidate solution vector into separate lists for
    phi values, t_init and delta_t values respectively.
    """
    l = len(chromosome)
    segments = int((l - 2)/2)

    phi_list = []
    for i in range(segments + 1):
        phi_list.append(chromosome[i])

    t_init = chromosome[segments + 1]
    delta_t_list = []

    for i in range(segments + 2, 2*segments + 2):
        delta_t_list.append(chromosome[i])
    t_list = [0]

    for i, delta_t in enumerate(delta_t_list):
        t_list.append(t_list[i] + delta_t)
    return(phi_list, t_init, t_list)


def interface(chromosome, crf_name, wet, sample_name, segments, algorithm):
    """
    This function serves as an interface between the Bayesian optimization,
    differential evolution, random search and grid search packages and the CRF
    function. This is necessary because the gradient profile specified by the
    candidate solution vector has to be transformed into a chromatogram
    (list of retention times and peak widths) for a given sample before the CRF
    score can be calculated. It does this using the following steps:

    1. Read in sample data using read_data.py
    2. Calculate retention times and peak widths for all sample compounds
       using the chromatographic simulator as implemented in retention_model.py
    3. Calculate and return the CRF score

    :chromosome: Gradient profile vector in the form of a numpy array.
    :crf_name: Name of the CRF to be used.
    :wet: Boolean indicating whether the wet or dry signal should be used.
    :sample_name: Name of the sample to be used.
    :segments: Number of segments in the gradient profile.
    :algorithm: Name of the algorithm that is calling the interface function.
    :return: CRF score for a chromatogram produced by the specified gradient
             profile.

    """

    phi_list, t_init, t_list = chromosome_to_lists(chromosome)

    # Get lnk0 and S data
    k0_list, S_list = rd.read_data(sample_name)
    #k0_list = [math.exp(lnk0) for lnk0 in lnk0_list]

    tR_list = []
    W_list = []

    if(wet == "True"):
        prefix = globals.results_folder + "/wet/" + crf_name + "/" + str(segments) + "segments/" + sample_name + "/" + algorithm + "/"
    else:
        prefix = globals.results_folder + "/dry/" + crf_name + "/" + str(segments) + "segments/" + sample_name + "/" + algorithm + "/"

    # Calculate retention times and peak widths
    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]

        tR, W = retention_model.retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)

    tlim = max(tR_list) + max(W_list)

    if(wet == "True" or crf_name in ["sum_of_kais", "prod_of_kais", "tyteca28", "tyteca35", "tyteca40"]):
        # Wet signal
        x, signal = pwd.create_signal(np.array(tR_list), np.array(W_list), tlim)
        tR_list, W_list, peak_heights, valleys, valley_height = pwd.detect_peaks(x, signal, height_thresh=0, plot=False, peak_info=True)

    # Calculate crf
    if(crf_name == "sum_of_res"):
        score = crf.capped_sum_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "prod_of_res"):
        score = crf.product_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "tyteca11"):
        score = crf.tyteca_eq_11(np.array(tR_list), np.array(W_list))
    elif(crf_name == "tyteca24"):
        score = crf.tyteca_eq_24(np.array(tR_list), np.array(W_list))
    elif(crf_name == "crf"):
        score = crf.crf(np.array(tR_list), np.array(W_list))
    elif crf_name == 'peak_purity':
        score = crf.peak_purity(tR_list, W_list)
    elif(crf_name == "sum_of_kais"):
        score =  crf.sum_of_kaiser(tR_list, peak_heights, valley_height)
    elif(crf_name == "prod_of_kais"):
        score =  crf.prod_of_kaiser(tR_list, peak_heights, valley_height)
    elif(crf_name == "tyteca28"):
        score = crf.tyteca_eq_28(tR_list, peak_heights,  valley_height)
    elif(crf_name == "tyteca35"):
        score = crf.tyteca_eq_35(tR_list, peak_heights,  valley_height, deadtime=t_0)
    elif(crf_name == "tyteca40"):
        score = crf.tyteca_eq_40(tR_list, peak_heights, valley_height)
    elif(crf_name == "time_info"):
        min_t, max_t = crf.time_info(tR_list, W_list)
        return min_t, max_t

    counter = get_counter()
    if counter < globals.popsize and algorithm in ["diffevo"]:
        # write score to file
        with open(prefix + "scores_init.txt", "a", newline="\n") as f:
            writer = csv.writer(f)
            # do not write row, but keep writing with a comma
            writer.writerow([score])
        # write gradient profile to file
        with open(prefix + "profiles_init.txt", "a", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(chromosome)
    set_counter(counter + 1)
    return(-1 * score)


# Input has to be a numpy array
def interface_pygad(crf_name, sample, wet, chromosome, segments, algorithm, solution_id):
    """
    This function serves as an interface between the genetic algorithm package
    and the CRF function. This is necessary because the gradient profile
    specified by the candidate solution vector has to be transformed into a
    chromatogram (list of retention times and peak widths) for a given sample
    before the CRF score can be calculated. It does this using the following
    steps:

    1. Read in sample data using read_data.py
    2. Calculate retention times and peak widths for all sample compounds
       using the chromatographic simulator as implemented in retention_model.py
    3. Calculate and return the CRF score

    :chromosome: Gradient profile vector in the form of a numpy array.
    :solution_id: Solution ID required by PyGAD package.
    :return: CRF score for a chromatogram produced by the specified gradient
             profile.

    """

    phi_list, t_init, t_list = chromosome_to_lists(chromosome)

    # Get lnk0 and S data
    k0_list, S_list = rd.read_data(sample)
    #k0_list = [math.exp(lnk0) for lnk0 in lnk0_list]

    tR_list = []
    W_list = []

    if(wet == "True"):
        prefix = globals.results_folder + "/wet/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"
    else:
        prefix = globals.results_folder + "/dry/" + crf_name + "/" + str(segments) + "segments/" + sample + "/" + algorithm + "/"

    # Calculate retention times and peak widths
    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]

        tR, W = retention_model.retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)

    tlim = max(tR_list) + max(W_list)

    if (wet == "True" or crf_name in ["sum_of_kais", "prod_of_kais", "tyteca28", "tyteca35", "tyteca40"]):
        # Wet signal
        x, signal = pwd.create_signal(np.array(tR_list), np.array(W_list), tlim)
        tR_list, W_list, peak_heights, valleys, valley_height = pwd.detect_peaks(x, signal, height_thresh=0, plot=False,
                                                                                 peak_info=True)

    # Calculate crf

    if(crf_name == "sum_of_res"):
        score = crf.capped_sum_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "prod_of_res"):
        score = crf.product_of_resolutions(np.array(tR_list), np.array(W_list), phi_list)
    elif(crf_name == "tyteca11"):
        score = crf.tyteca_eq_11(np.array(tR_list), np.array(W_list))
    elif(crf_name == "tyteca24"):
        score = crf.tyteca_eq_24(np.array(tR_list), np.array(W_list))
    elif(crf_name == "crf"):
        score = crf.crf(np.array(tR_list), np.array(W_list))
    elif crf_name == 'peak_purity':
        score = crf.peak_purity(tR_list, W_list)
    elif (crf_name == "sum_of_kais"):
        score = crf.sum_of_kaiser(tR_list, peak_heights, valley_height)
    elif (crf_name == "prod_of_kais"):
        score = crf.prod_of_kaiser(tR_list, peak_heights, valley_height)
    elif (crf_name == "tyteca28"):
        score = crf.tyteca_eq_28(tR_list, peak_heights, valley_height)
    elif (crf_name == "tyteca35"):
        score = crf.tyteca_eq_35(tR_list, peak_heights, valley_height, deadtime=t_0)
    elif (crf_name == "tyteca40"):
        score = crf.tyteca_eq_40(tR_list, peak_heights, valley_height)
    elif (crf_name == "time_info"):
        min_t, max_t = crf.time_info(tR_list, W_list)
        return min_t, max_t
    counter = get_counter()
    if counter < globals.popsize and algorithm in ["genalgo"]:
        # write score to file
        with open(prefix + "scores_init.txt", "a", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow([score])
        # write gradient profile to file
        with open(prefix + "profiles_init.txt", "a", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(chromosome)
    set_counter(counter + 1)
    return(score)

def set_counter(n):
    # write n to a file
    with open('counter.txt', 'w') as file:
        file.write(str(n))

def get_counter():
    # read n from a file
    with open('counter.txt', 'r') as file:
        loaded_data = int(file.read())
    return loaded_data

