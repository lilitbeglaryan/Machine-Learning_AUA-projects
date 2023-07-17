[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10126831&assignment_repo_type=AssignmentRepo)
# Instructions

⚠️

* **Do not modify the `tests` folder! It may cause errors and hence a decrease in your grading.**

* **Do not modify function names, variable names, or important code logic if you're not instructed to do so in README or
  directly in code comments or docstring. It may cost you a decrease in your grade.**

* **Add, modify or delete only those code parts which are either instructed by README, comment, or docstring,
  or they are intentionally left blank (added placeholders `pass`, `...`, `None`) for you to fill.**

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
Classification
**Points: 100**

In this homework, you are required to apply the following algorithms to the
[MNIST handwritten dataset from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits):

* [Decision trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
* [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Simple bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
* [Boosting algorithms](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

You are also required to perform cross-validation and grid search techniques. Feel free to use any library you want,
but sklearn contains most of your needs.

Please make sure to visualize the dataset, perform some exploratory data analysis (EDA), visualize decision trees,
and visualize decision boundaries.

Write your utility functions in separate `.py` files, but perform your experiments and report your findings in a
jupyter notebook.

At the end of your experiments, you must report your classification metrics and explain why you chose certain metrics.
Your report should be constructed in an understandable way.
Compare your models, discuss results, explain why one model is better than the other. 

Please take note, that this homework is for checking your knowledge of classification, focus on that more than on
calling `tree.fit()`. 
This homework doesn't have any tests. 