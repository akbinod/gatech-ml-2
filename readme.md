<h2>Georgia Tech : CS-7641 - Machine Learning : Spring 2020 </h1>
<h2>Assignment 2 - Randomized Optimization</h3>

<h3>Running the code</h3>
<p>
Running main.py will run the analysis that forms the bulk of this assignment. Output consists of plots (pasted into the report) and run stats which have been put into macOS .numbers files - one for each of the problems solved.


Everything is kicked off from main.py. Please change the function called at the bottom of the file. You can choose to run one of 3 main problems, or run the ANN (part 2 of the assignment)


</p>
<h3>Code Environment</h3>
<p>
This code was developed on a mac, using VSCode, and python 3.7.6. Where file paths are involved, you might need to tweak things just a bit based on how you run it, and your file system. All of that is possible within main.py

</p>
<h3>Dependencies</h3>
<ul>
<li>numpy
<li>matplotlib
<li>sklearn(0.22)
<li>PyTorch (1.3.1)
<li>mlrose_hiive
</ul>

<h3>Code Organization</h3>
<p>
The various Solver and Algorithm classes implement all the code required by this assignment. Code for Assignment 1 has been left in, I did not have the time to examine what was safe to remove. The files to review are:

BaseSolver (base class for implementing a solution to one of the problems)
	Queens.py
	FourPeaks.py
	Knapsack.py
BaseAlgorithm (base class for implementing an algorighm)
	SimulatedAnnealing
	RandomHillClimbing
	GeneticAlgorithm
	Mimic

Apart from these pieces, there are a number of utility files for anciliary operations like plotting, timing, etc which have little bearing on the main purpose of this project, but help me profile and debug. This project builds on code that I developed during CS-7642 during  Fall '19.
</p>


