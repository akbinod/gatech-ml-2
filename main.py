# import appnope
import json
import numpy as np
import setproctitle
import os
import sys

# from akbinod.Utils.Plotting import Plot

from learners import BaseLearner, RandomForest, LearnerParams, SVM
from learners import DecisionTree, KNearestNeighbors, BoostedTree, NeuralNetwork
from learners.constants import LearnerMode

from Solvers import Queens, FourPeaks, Knapsack, SolverParams, IrisANN

SEED = 0

def process_files(paths, target):
	results = {}
	plots = []
	for data_path in paths:
		results[data_path] = {}
		params = LearnerParams(LearnerMode.classification, learning_target=target, data_path=data_path, cv = 5)
		results[data_path] = {}

		# lrnr = DecisionTree(params)
		# lrnr = BoostedTree(params)
		# lrnr = KNearestNeighbors(params)
		# lrnr = SVM(params)
		lrnr = NeuralNetwork(params)

		lrnr.train()
		res = lrnr.infer()
		results[data_path]['train'] = res[0]
		results[data_path]['test'] = res[1]
		results[data_path]['best_params'] = lrnr.model.best_params_
		plots.append(lrnr.plot_learning_curve())

	print(json.dumps(results))
	for plt in plots:
		plt.show()

def get_coffee_files():

	paths = []
	paths.append('./data/flavor-coffee.csv')
	paths.append('./data/flavor-coffee-no-altitude.csv')
	paths.append('./data/flavor-coffee-no-country-altitude.csv')
	paths.append('./data/flavor-coffee-no-region-altitude.csv')
	paths.append('./data/flavor-coffee-no-region-altitude-elevation.csv')
	return paths

def get_iris_files():
	return ['./data/iris-1.csv']

def get_wine_files():
	return ['./data/winequality.csv']

def main_a1(get_coffee):
	# files = get_wine_files()

	if get_coffee:
		files = get_coffee_files()
		process_files(files, "target")
	else:
		files = get_iris_files()
		process_files(files, "Species")

def SolveQueens(tune = False):
	sp = SolverParams()
	sp.num_queens = 16

	slv = Queens(sp)
	if tune:
		slv.tune()
	else:
		slv.solve(runs=20)
		plt = slv.plot_comparisons()
		plt.show()

def SolveKnapsack(tune = False):
	sp = SolverParams()
	sp.items = 10
	slv = Knapsack(sp)
	if tune:
		slv.tune()
	else:
		slv.solve(runs=20)
		plt = slv.plot_comparisons()
		plt.show()

def SolvePeaks(tune = False):
	sp = SolverParams()
	sp.length = 32

	slv = FourPeaks(sp)
	if tune:
		slv.tune()
	else:
		slv.solve(runs=20)
		plt = slv.plot_comparisons()
		plt.show()

def part2():
	sp = SolverParams()

	slv = IrisANN(sp)
	slv.benchmark()

def main(run_name = ""):
	if run_name == "":
		# just use the folder name - that's probably going to be the project/run
		run_name = os.path.dirname(os.path.curdir)

	# this does not show up in mac activity monitor
	# not in top either - that just showed Python
	setproctitle.setproctitle (run_name)
	print(setproctitle.getproctitle())

	# with appnope.nope_scope():
	# SolveQueens()
	# SolvePeaks
	# SolveKnapsack

	part2()

if __name__ == "__main__":
	# change this to something that shows the grid search your in
	main("devel project")