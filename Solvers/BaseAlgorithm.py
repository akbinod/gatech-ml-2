import numpy as np
from matplotlib import pyplot as plt

class BaseAlgorithm():
	def __init__(self, params):
		self.params = params

		self.fitness_scores = []
		self.iteration_scores = []
		self.best_state = None
		self.best_fitness = None

	def __str__(self):
		ret = str(self.name)
		# if self.best_fitness is not None:
		# 	ret = ret + " - best: " + str(self.best_fitness)

		return ret

	def tune(self, problem, init_state):
		# Implementors must override this function.
		# Must return best_state, best_fitness,
		# iteration_scores[] (same as curve returned by the algorithm),
		# fitness_scores[] (accumulation of fitness scores)
		raise NotImplementedError()

	def solve(self, problem, init_state):
		# Implementors must override this function.
		# Must return best_state, best_fitness,
		# iteration_scores[] (same as curve returned by the algorithm),
		raise NotImplementedError()

	def plot_tuning(self):
		"""
	state, fitness, it, fc, = slv.solve()
	plt = Plot("Fitness over iterations",'iterations', 'fitness',it)
	plt.show()
	plt = Plot("Fitness function calls",'fitness call num', 'fitness',fc)
	plt.show()


		Generate 2 plots:
			the scores over iterations curve
			, the scores over fitness function calls curve


		axes : array of 2 axes, optional (default=None)
			Axes to use for plotting the curves.

		ylim : tuple, shape (ymin, ymax), optional
			Defines minimum and maximum yvalues plotted.


		train_sizes : array-like, shape (n_ticks,), dtype float or int
			Relative or absolute numbers of training examples that will be used to
			generate the learning curve. If the dtype is float, it is regarded as a
			fraction of the maximum size of the training set (that is determined
			by the selected validation method), i.e. it has to be within (0, 1].
			Otherwise it is interpreted as absolute sizes of the training sets.
			Note that for classification the number of samples usually have to
			be big enough to contain at least one sample from each class.
			(default: np.linspace(0.1, 1.0, 5))
		"""

		# fitness_mean = np.mean(self.fitness_scores, axis=0)
		# fitness_std = np.std(self.fitness_scores, axis=0)

		# iterations_mean = np.mean(self.iteration_scores, axis=0)
		# iterations_std = np.std(self.iteration_scores, axis=0)

		_, axes = plt.subplots(1, 2, figsize=(20, 5))

		# Plot fitness over iterations
		axes[0].set_title("Fitness/Iterations: " + str(self))
		axes[0].set_xlabel("iterations")
		axes[0].set_ylabel(self.fitness_label)

		# ylim=(0.2, 1.01)
		# axes[0].set_ylim(*ylim)
		# axes[0].grid()
		# axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
		# 					train_scores_mean + train_scores_std, alpha=0.1,
		# 					color="r")
		# axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
		# 					test_scores_mean + test_scores_std, alpha=0.1,
		# 					color="g")
		if self.iteration_scores_sa is not None and len(self.iteration_scores_sa):
			axes[0].plot(self.iteration_scores_sa, '-', color="r", label="sa" + str(self.best_fitness_sa))

		axes[0].legend(loc="best")

		# Plot fitness over fitness function calls
		axes[1].set_title("Fitness/Fitness Function Calls: " + str(self))
		axes[1].set_xlabel("fitness function calls")
		axes[1].set_ylabel(self.fitness_label)
		if self.fitness_scores_sa is not None and len(self.fitness_scores_sa):
			axes[1].plot(self.fitness_scores, '-', scalex=True, scaley=True, color="r", label="sa-" + str(self.best_fitness_sa))

		axes[1].legend(loc="best")

		# axes[1].grid()
		# axes[1].plot(train_sizes, fit_times_mean, 'o-')
		# axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
		# 					fit_times_mean + fit_times_std, alpha=0.1)


		return plt
