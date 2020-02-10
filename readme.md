<h2>Georgia Tech : CS-7641 - Machine Learning : Spring 2020 </h1>
<h2>Assignment 1 - Supervised Learning</h3>

<h3>Running the code</h3>
<p>
Running main.py will run the analysis that forms the bulk of this assignment. The data used for this assignment is in the ./data folder. These are csv files containing the complete data.


Everything is kicked off from main.py. As last checked in, this should analyse the Iris dataset. To analyse the coffee dataset, edit the last line at the bottom of the file, and switch the flag to True.

</p>
<b>Tweaking Runs/Changing the analysis</b>
<p>
This assignment conducts 5 different analysis on data (Decision Trees, Boosted Decision Trees, KNN, SVM, Neural Network). The function process_files() controls which analysis is run when you execute main.py. The last checked in version runs the Decision Tree analysis. Uncomment the line that instantiates the appropriate learner to switch to a different analysis.

To change a hyper parameter, you would need to edit the build_pipeline() method of a learner. Learners differ from each other mainly in the scikit class they instantiate, and the hyperparameters they specify in the call to GridSearchCV().

The bulk of the code that invokes sklearn is in the base class - BaseLearner. This is where the train and infer code is located.

</p>
<h3>Code Environment</h3>
<p>
This code was developed on a mac, using VSCode, and python 3.7.6. Where file paths are involved, you might need to tweak things just a bit based on how you run it, and your file system. All of that is possible within main.py

</p>
<h3>Dependencies</h3>
<h4>General</h4>
<ul>
<li>numpy
<li>matplotlib
<li>sklearn(0.22)
<li>PyTorch (1.3.1)
</ul>

<h3>Code Organization</h3>
<p>
The various Learner classes implement all the code required by this assignment.

Apart from these pieces, there are a number of utility files for anciliary operations like plotting, timing, etc which have little bearing on the main purpose of this project, but help me profile and debug. This project builds on code that I developed during CS-7642 during  Fall '19.
</p>


