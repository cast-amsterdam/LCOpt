# LC Gradient Profile Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oaOq1I8jeYaumZOiFN8RAz8r9nK6EZNJ?usp=sharing)

This directory contains Python code for creating computer simulations of chromatographic experiments and applying optimization algorithms to gradient profile optimization.

To cite:
TODO

## Installation

Use the package and environment management system  [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the conda environment necessary to run the code in this directory from the gpo_environment.yml file.

```bash
conda env create --file install_environment.yml
conda activate lcopt_env
```

## Usage
We here provide command line usage of the code. We provide other ways of using this package in a Google Collab.
### Specifying a Sample

The filepath to the sample to be separated must be specified manually in _globals.py_. The sample must be specified in a .csv file where each row represents a single compound. The first column should contain the compound's k<sub>0</sub> value and the second column should contain the compound's S value. For examples, see the './samples/' folder.

A sample you can use for testing purposes can be found at './samples/snyder_samples/regular_sample.csv'. This is a sample taken from the book _High-Performance Gradient Elution_ by Lloyd R. Snyder and John W. Dolan. It is a sample consisting of 9 herbicides:

| Compound | k<sub>0</sub> | S |
|----------|:-------------:|------:|
| Simazine | 9.650 | 3.41 |
| Monolinuron | 11.623 | 3.65 |
| Metobromuron | 13.504 | 3.746 |
| Diuron | 16.710 | 3.891 |
| Propazine | 24.804 | 4.222 |
| Chloroxuron | 36.671 | 4.636 |
| Neburon | 50.400 | 4.882 |
| Prometryn | 113.410 | 5.546 |
| Terbutryn | 177.328 | 5.914 |


### Chromatographic Response Functions

We support many different Chromatographic Response Functions (CRFs) that can be used to evaluate the quality of a gradient profile.
These can be found in the _crf.py_ file.


### Running an Optimization Algorithm
All optimization algorithms can be run standalone from the command line, and largely follow the same logic. The following sections describe how to run each algorithm.

##### Bayesian Optimization

```bash
python bayesopt.py <crf> <sample> <dry/wet> <iterations> <segments>
# example
python bayesopt.py sum_of_res sample_real True 10 2
```
Replace **sample** in the command with the name of the sample you want to separate. Replace **crf** with the name of the CRF you want to use. Replace **dry/wet** with either 'True' or 'False' depending on whether you want to use the wet or dry mode of peak detection.
In **dry**, the predicted retention times and peak widths are directly use in the computation of the crf. 
In **wet**, a 1D signal is generated from the predicted retention times and peak widths, then peak detection is performed on this signal to find the retention times and peak widths that are used in the computation of the crf. 
Replace **iterations** in the command with the number of iterations you want to algorithm to run for. Replace **segments** with the number of segments you want the gradient profile to consist of.

The runtime and the best solution that was found and its CRF score will be printed to the terminal. The virtual chromatogram produced by the best found solution will also be plotted.

##### Differential Evolution

```bash
python diffevo.py <crf> <sample> <dry/wet> <iterations> <segments>
```

##### Genetic Algorithm

```bash
python ga.py <crf> <sample> <dry/wet> <iterations> <segments>
```

##### CMA-ES

```bash
python cmaes.py <crf> <sample> <dry/wet> <iterations> <segments>
```


##### Random Search

```bash
python random_search.py <crf> <sample> <dry/wet> <iterations> <segments>
```

##### Grid Search

```bash
python grid_search.py <crf> <sample> <dry/wet> <grid_points> <segments>
```
Replace **grid_points** in the command with the number of grid points you want in the grid for each dimension. Note that the total number of grid points is (grid_points^(2 x segments + 2)), and will result in a large grid rather quickly for a large number of segments.
Replace **segments** with the number of segments you want the gradient profile to consist of.

The runtime and the best solution that was found and its CRF score will be printed to the terminal. The virtual chromatogram produced by the best found solution will also be plotted.


### Creating a chromatogram from a manually specified gradient profile

To produce a virtual chromatogram from a manually specified gradient profile, first set the HPLC system parameters in _globals.py_. Then, use the following command to produce the chromatogram:

```bash
python chromatogram_given_profile.py <phi_list> <t_init> <t_list> <sample>
```

Replace **phi_list** with a list of phi values; one for each turning point in the gradient profile. Replace **t_list** with a list of t values; one for each turning point in the gradient profile.

The following image shows an example of a virtual chromatogram of the sample of 9 herbicides for a given gradient profile. In this case, the 2-segment gradient profile was specified by **phi_values** = "[0.35, 0.4, 0.8]" and **t_values**" = "[0, 5, 10]". With an initial time of 1 minute.

![image1](/images/ex_chromatogram.png)

```bash
python chromatogram_given_profile.py '[0.35,  0.4, 0.8]' 1 '[0,  5, 10]' regular_sample
```

### Running a Meta Experiment
A meta experiment allows you to run an optimization algorithm with given settings multiple times and to save the results of each run. This is useful for comparing the performance of different optimization algorithms.
This can be done for each optimization algorithm listed above by running the following command:

```bash
python meta_experiment.py <optimization algorithm> <segments> <repeats> <iterations/generations/grid points> <sample> <dry/wet> <crf>
```
Replace **optimization algorithm** with one of the following: GenAlgo, BayesOpt, RandomSearch, DiffEvo, CMA or GridSearch. Replace **segments** with the number of segments you want the gradient profile to consist of. Replace **repeats** with the number of times you want to run the optimization algorithm. Replace **iterations/generations/grid points** with the number of iterations/generations/grid points you want the algorithm to run for. Replace **sample** with the name of the sample you want to separate. 
Replace **dry/wet** with either 'True' or 'False' depending on whether you want to use the wet or dry mode of peak detection. Replace **crf** with the name of the CRF you want to use.

Replace iterations for the number function evaluations you want to perform when using BayesOpt, CMA, or RandomSearch.
In case of DiffEvo, and EA, specify the number of generations, here the number of function evaluations is equal to the number of generations times the population size.
In case of GridSearch, specify the number of grid points you want in the grid for each dimension. Note that the total number of grid points is (grid_points^(2 x segments + 2)), and will result in a large grid rather quickly for a large number of segments.

Running _meta_experiment.py_ will write results to the directory specified in _globals.py_ and will store files in the following structure:

```
/results/<dry/wet>/<crf>/<segments>/<sample>/<optimization algorithm>/
```

and will store the following files:

* **score.csv** - contains a line for each repeat of the optimization algorithm. Each line contains the score of the best solution found in that repeat.
* **purity.csv** - Containing three columns with the peak purity, time of the first eluting compound, and time of the last eluting compound for the best solution found in each repeat.
* **runtime.csv** - contains a line for each trial, with the cumulative runtime for each iteration. For the total runtime, take the last element.
* **solutions.csv** - A csv file containing the best solution found in each repeat of the optimization algorithm.
* **all_solutions_<repeat>.csv** - A csv file containing all solutions found in each repeat of the optimization algorithm.
* **best_solutions_chrom_<repeat>.png** - A plot of the best solution found in each repeat of the optimization algorithm.

Note that GenAlgo and DiffEvo store these values per generation instead of per function evaluation. For instance, for a population size of 10, and 2 generations, **score.csv**
would contain 3 values per line, the 0th generation, the 1st generation, and the 2nd generation. The 0th generation contains the score of the best solution found in the initial population.

Also note that GridSearch looks slightly different:
Since GridSearch is a deterministic method, there is no notion of repeats. Therefore, the files are stored in the following structure:

* **score.csv** - contains the best value found for the grid search.
* **purity.csv** - Containing three columns with the peak purity, time of the first eluting compound, and time of the last eluting compound for the best solution found in the grid search.
* **runtime.csv** - contains the runtime of the grid search.
* **solution.csv** - A csv file containing the best solution found in the grid search.
* **all_solutions_0.csv** - A csv file containing all solutions found in the grid search.
* **best_solutions_chrom.png** - A plot of the best solution found in the grid search.

See the _jobscripts_ folder for examples of how to run a list of meta experiments on a HPC cluster.