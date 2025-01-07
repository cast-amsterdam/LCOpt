# write a testing_brute.py script that will run the program with the optimizers standalone, and meta experiment

# all algos standalone
python random_search.py sum_of_res sample_real True 10 4
python bayesopt.py sum_of_res sample_real True 10 4
python ga.py sum_of_res sample_real True 1 4
python diffevo.py sum_of_res sample_real True 1 4
python cmaes.py sum_of_res sample_real True 1 4
python grid_search.py sum_of_res sample_real True 1 4

# all algos with meta experiment
python meta_experiment.py RandomSearch 4 2 10 sample1 False sum_of_res
python meta_experiment.py BayesOpt 4 2 10 sample1 False sum_of_res
python meta_experiment.py GenAlgo 4 2 1 sample1 False sum_of_res
python meta_experiment.py DiffEvo 4 2 1 sample1 False sum_of_res
python meta_experiment.py CMA 4 2 1 sample1 False sum_of_res
python meta_experiment.py GridSearch 4 1 2 sample1 False sum_of_res