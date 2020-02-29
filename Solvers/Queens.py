import mlrose_hiive as mlrose
import numpy as np
from Solvers.BaseSolver import BaseSolver
from Solvers.SimulatedAnnealing import SimulatedAnnealing
from Solvers.RandomHIllClimbing import RandomHIllClimbing

import copy

class Queens(BaseSolver):
	def __init__(self, params, auto_set_hp = True):
		super().__init__(params, auto_set_hp)

		self.name = "Queens"
		# set this to True if using the custom maximizing function
		self.maximize = False
		# change this as well if using the custom maximizing function
		self._fitness_label = "queen attacks"

		self.sa_params = None
		self.rhc_params = None
		self.ga_params = None
		self.mimic_params = None

		if self.auto_set_hp:
			self.init_empirical_hp()

		# Define initial state
		self.init_state = np.arange(start=0, stop=self.params.num_queens, step=1)
		# self.init_state = None

		# Define the problem, and initialize custom fitness function object
		# This delegates to the builtin but gives us the opportunity
		# to capture fitness values as they are generated. For use in comparison.
		self.fitness_delegate = mlrose.Queens() #will be used by custom fitness fn
		self.problem = mlrose.DiscreteOpt(length = self.params.num_queens
										, fitness_fn = mlrose.CustomFitness(self.fitness_fn)
										, maximize = self.maximize
										, max_val = self.params.num_queens)

	def init_empirical_hp(self):
		# these truths we hold to be self evident (come from HP tuning)

		# set up whatever we know about SA
		self.sa_params = copy.deepcopy(self.params)
		self.sa_params.random_state = 1
		self.sa_params.decay_schedule = mlrose.GeomDecay()
		self.sa_params.max_iters = np.inf
		self.sa_params.max_attempts = 2500

		# set up whatever we know about RHC
		self.rhc_params = copy.deepcopy(self.params)
		self.rhc_params.random_state = 1
		self.rhc_params.restarts = 25
		self.rhc_params.max_iters = np.inf
		self.rhc_params.max_attempts = 2500

	def tune(self):
		# alg = SimulatedAnnealing(self.params)
		# # so that the fitness function records to the correct array
		# self.current_fitness_score = alg.fitness_scores
		# alg.tune(self.problem, self.init_state, False)

		alg = RandomHIllClimbing(self.params)
		# so that the fitness function records to the correct array
		self.current_fitness_score = alg.fitness_scores
		alg.tune(self.problem, self.init_state, False)

		return

	def solve(self):
		# Solve problem using simulated annealing
		if self.sa_params is not None:
			pa = self.sa_params
		else:
			pa = self.params
		# self.algorithms.append(SimulatedAnnealing(pa))
		self.algorithms.append(RandomHIllClimbing(pa))

		for alg in self.algorithms:
			# so that the fitness function records to the correct array
			self.current_fitness_score = alg.fitness_scores
			alg.solve(self.problem, self.init_state)
			if self.maximize:
				alg.fitness_scores = [- score for i, score in enumerate(alg.fitness_scores)]
				alg.iteration_scores = [- score for i, score in enumerate(alg.iteration_scores)]

		return

	# def fitness_fn(self, state):
	# 	self._fitness_label = "queens correctly placed"
	# # Initialize counter
	# 	bad_queens = 0

	# 	for i in range(len(state) - 1):
	# 		# For each queen
	# 		for j in range(i + 1, len(state)):
	# 			# Check for horizontal, diagonal-up and diagonal-down attacks
	# 			if (state[j] == state[i]) or (state[j] == state[i] + (j - i)) or (state[j] == state[i] - (j - i)):
	# 				# attack exists, increment counter of misplaced queens
	# 				bad_queens += 1
	# 				break
	# 	good_queens = self.solver_params.num_queens - bad_queens
	# 	# add the latest score to the list to be returned
	# 	self.current_fitness_score.append(good_queens)
	# 	return good_queens

	def fitness_fn(self, state):

		# delegate to the real fitness function
		ret =  self.fitness_delegate.evaluate(state)
		# record the score for reporting
		self.current_fitness_score.append(ret)
		# to be consumed by mlrose
		return ret

	@property
	def fitness_label(self):
		return self._fitness_label

	def __str__(self):
		return str(self.params.num_queens) + " Queens"