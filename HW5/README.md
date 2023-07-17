[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/dcynn1Wn)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10711576&assignment_repo_type=AssignmentRepo)
# Instructions

⚠️

* **Do not modify the `tests` folder! It may cause errors and hence a decrease in your grading.**

* **Do not modify function names, variable names, or important code logic if you're not instructed to do so in README or
  directly in code comments or docstring. It may cost you a decrease in your grade.**

* **Add, modify or delete only those code parts which are either instructed by README, comment, or docstring,
  or they are intentionally left blank (added placeholders `pass`, `...`, `None`, `NotImplementedError`) for you to
  fill.**

* **Problem specific instructions are done via comments, read them, also pay more attention on TODO comments**

*Functions usually return values, if the instructor wants you to print something, it will directly be instructed in the
code, either by comment, or placeholder, or there will be a `print` function in the code*

### Recommendation

It's recommended to use different virtual environments for different projects (HWs).
You can always find list of [required libraries](requirements.txt) in your HW directory.
You may install requirements before solving your projects, it will increase your chances for having working code:

```shell
pip install -r requirements.txt
```

Happy coding 🧑‍💻.

# Problem statements

## Problem 1

Wine Clustering **POINTS 100**

In this assignment, you will apply Gaussian Mixture Models (GMMs), k-means, and k-medoids clustering algorithms on
the [Wine dataset](https://drive.google.com/file/d/1NgL_wUQZpwLzTEYWSOamgSjZwXHRLmDc/view?usp=share_link) and compare
their performance.

About the data:

The following descriptions are adapted from the UCI webpage:

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from N
different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of
wines.

The attributes are:

    Alcohol
    Malic acid
    Ash
    Alcalinity of ash
    Magnesium
    Total phenols
    Flavanoids
    Nonflavanoid phenols
    Proanthocyanins
    Color intensity
    Hue
    OD280/OD315 of diluted wines
    Proline

You can use other modules from imported libraries, and you may also use pandas, seaborn, and any other library that you
find necessary.

Write your utility functions in [utils](utils.py), do your final analysis on [Clustering notebook](Clustering.ipynb).
Don't forget to run all your notebook before pushing to GitHub to properly show all plots.

1. Load the Wine dataset and explore features, plot features and do EDA. **POINTS: 6**

2. Apply GMMs, k-means, and k-medoids clustering algorithms on the Wine dataset. Use the scikit-learn library to
   implement these algorithms.

3. Experiment with data feature scaling for all three algorithms, resulting at least six kind of experiments (3 models,
   scaling vs no-scaling).
   You can choose any kind of scaling that you find useful for your features. **Points: 24**

3. Determine the optimal number of clusters for k-means and k-medoids using the silhouette or other score. Evaluate the
   clustering results using silhouette or other scores. Compare the scores to determine which algorithm performs better
   on the
   Wine dataset. **POINTS: 10**

4. Fit GMMs with different numbers of components (clusters) to the Wine dataset. Use grid search and the Bayesian
   Information Criterion (BIC) to determine the optimal number of clusters. **POINTS: 10**

5. Plot the BIC values for each model and analyze the results to choose the best model. Discuss the advantages and
   disadvantages of using BIC for model selection and the implications of your choice for the Wine dataset. **POINTS:
   10**
6. Visualize the cluster assignments produced by each algorithm and analyze the results (you can choose 0 and 1
   features, or any pair of features which are visually more informative). Discuss the strengths and
   weaknesses of each clustering algorithm (3 of them) in the context of the Wine dataset. **POINTS: 20**

7. Your code quality, readability, report notebook structure, and readability will be checked and graded. **POINTS: 20**

