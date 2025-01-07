import csv
import json
import random
import numpy as np
import globals

prefix = globals.sample_path

def read_data(sample_name):
    """
    Return list of k0 and S samples for a given sample. Sample needs to be
    manually specified here, because optimization algorithm packages don't allow
    for extra arguments in the objective function, other than the parameters
    to be optimized.

    """

    k0_list = []
    S_list = []

    # with open('globals.json') as json_file:
    #     variables = json.load(json_file)
    #
    # sample_name = str(variables["sample_name"])

    path = prefix + sample_name + ".csv"

    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            k0_list.append(float(row[0]))
            S_list.append(float(row[1]))

    if 'real' in sample_name:
        return (k0_list, S_list)
    if 'dist' not in sample_name:
        # Get the first 35 elements from each list. This is the sample.
        k0_list = k0_list[:35]
        S_list = S_list[:35]

    return(k0_list, S_list)
