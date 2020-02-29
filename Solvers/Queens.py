import mlrose_hiive as mlrose
import numpy as np
from Solvers.BaseSolver import BaseSolver
from Solvers.RandomHIllClimbing import RandomHillClimbing
from Solvers.SimulatedAnnealing import SimulatedAnnealing
from Solvers.GeneticAlgorithm import GeneticAlgorithm
from Solvers.Mimic import Mimic

from akbinod.Utils.TimedFunction import TimedFunction
# import Solvers.SimulatedAnnealing


import copy

class Queens(BaseSolver):
	def __init__(self, params):
		super().__init__(params)

		self.name = "Queens"
		# set this to True if using the custom maximizing function
		self.maximize = False
		# change this as well if using the custom maximizing function
		self._fitness_label = "queen attacks"

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
		# at the 16 Queens level
		# set up whatever we know about SA - good, done
		self.sa_params = copy.deepcopy(self.params)
		self.sa_params.decay_schedule = mlrose.GeomDecay()
		self.sa_params.random_state = None
		self.sa_params.max_iters = 2500
		self.sa_params.max_attempts = 250

		# set up whatever we know about RHC - in progress
		self.rhc_params = copy.deepcopy(self.params)
		self.rhc_params.random_state = None
		self.rhc_params.restarts = 250
		self.rhc_params.max_iters = 250
		self.rhc_params.max_attempts = 250

		# set up whatever we know about GA
		self.ga_params = copy.deepcopy(self.params)
		self.ga_params.pop_size = 1000
		self.ga_params.pop_breed_percent = 0.50
		self.ga_params.mutation_prob = 0.25
		self.rhc_params.max_iters = np.inf
		self.rhc_params.max_attempts = 250

		# set up whatever we know about Mimic
		self.mimic_params = copy.deepcopy(self.params)
		self.mimic_params.pop_size = 600
		self.mimic_params.keep_pct = 0.1
		self.mimic_params.noise = 0.01
		self.mimic_params.max_iters = np.inf
		self.mimic_params.max_attempts = 20

	@TimedFunction(True)
	def tune(self):
		alg = SimulatedAnnealing(self.params)
		# so that the fitness function records to the correct array
		self.current_fitness_score = alg.fitness_scores
		alg.tune(self.problem, self.init_state, False)

		alg = RandomHillClimbing(self.params)
		# so that the fitness function records to the correct array
		self.current_fitness_score = alg.fitness_scores
		alg.tune(self.problem, self.init_state, False)

		alg = GeneticAlgorithm(self.params)
		# so that the fitness function records to the correct array
		self.current_fitness_score = alg.fitness_scores
		alg.tune(self.problem, self.init_state, False)

		alg = Mimic(self.params)
		# so that the fitness function records to the correct array
		self.current_fitness_score = alg.fitness_scores
		alg.tune(self.problem, self.init_state, False)

		return
	def solve(self):
		# Solve problem using all the tuned algorithms that apply to this proble,
		pa = self.params if self.sa_params is None else self.sa_params
		self.algorithms.append(SimulatedAnnealing(pa))

		pa = self.params if self.rhc_params is None else self.rhc_params
		self.algorithms.append(RandomHillClimbing(pa))

		# pa = self.params if self.ga_params is None else self.ga_params
		# self.algorithms.append(GeneticAlgorithm(pa))

		# pa = self.params if self.mimic_params is None else self.mimic_params
		# self.algorithms.append(Mimic(pa))

		return super().solve()

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