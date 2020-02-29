import numpy as np
from matplotlib import pyplot as plt

class BaseSolver():
	def __init__(self, params):
		self.params = params

		self.current_fitness_score = None
		self.algorithms = []
		self.plot_colors = ['red', 'blue', 'green', 'black']
		self.sa_params = None
		self.rhc_params = None
		self.ga_params = None
		self.mimic_params = None

	def solve(self):

		self.results = []
		for alg in self.algorithms:
			# so that the fitness function records to the correct array
			self.current_fitness_score = alg.fitness_scores
			alg.solve(self.problem, self.init_state)
			if not self.maximize:
				alg.fitness_scores = [- score for i, score in enumerate(alg.fitness_scores)]
				alg.iteration_scores = [- score for i, score in enumerate(alg.iteration_scores)]

			r = {}
			r["alg"] = str(alg)
			r["time"] = alg.solve_time
			r["best_fitness"] = alg.best_fitness
			r["average_score"] = round(float(np.mean(alg.iteration_scores)),4)
			r["iterations"] = len(alg.iteration_scores)
			r["fit_fn_calls"] = len(alg.fitness_scores)
			self.results.append(r)

		print(f"alg\tbest_f\tavg_f\titers\tfn_cal\ttime")
		for r in self.results:
			print(f"{r['alg']}\t{r['best_fitness']}\t{r['average_score']}\t{r['iterations']}\t{r['fit_fn_calls']}\t{r['time']}")

		return



	def fitness_fn(self):
		# implementors must override this function - even if the
		# override just calls the builtin function from it.
		# The key is to add the fitness score to an array.
		# Return this array from solve() - see above
		raise NotImplementedError()

	@property
	def fitness_label(self):
		raise NotImplementedError()

	def __str__(self):
		return self.name

	def plot_comparisons(self):
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
		axes[0].set_title("Benchmarking Fitness (iterations): " + str(self))
		axes[0].set_xlabel("iterations")
		axes[0].set_ylabel(self.fitness_label)

		for i, alg in enumerate(self.algorithms):
			if len(alg.iteration_scores):
				axes[0].plot(alg.iteration_scores, '-'
							, color=self.plot_colors[i]
							, label= str(alg) +  ": " + str(alg.best_fitness)
							)
		axes[0].legend(loc="best")

		# Plot fitness over fitness function calls
		axes[1].set_title("Benchmarking Fitness (fn calls): " + str(self))
		axes[1].set_xlabel("fitness function calls")
		axes[1].set_ylabel(self.fitness_label)
		for i, alg in enumerate(self.algorithms):
			if len(alg.fitness_scores):
				axes[1].plot(alg.fitness_scores, '-'
							, color=self.plot_colors[i]
							, label= str(alg) +  ": " + str(alg.best_fitness)
							)
		axes[1].legend(loc="best")

		# axes[1].grid()
		# axes[1].plot(train_sizes, fit_times_mean, 'o-')
		# axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
		# 					fit_times_mean + fit_times_std, alpha=0.1)


		return plt
