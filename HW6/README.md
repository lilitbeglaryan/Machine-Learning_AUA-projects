[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/bjglfevZ)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10933543&assignment_repo_type=AssignmentRepo)
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

### PCA for Image Classification on LFW People Dataset.  **Points 100**

In this homework, you will apply Principal Component Analysis (PCA) to the task of image classification using the LFW
People dataset. You will determine the optimal number of principal components to use, train a classifier, and compare
the results with different numbers of components and the full set of features. Your main goal is to analyze the
trade-offs between component number, model performance, and computation time.

**Task 1: Load and preprocess the dataset**. **Points 10**

1. Load the LFW People dataset (`fetch_lfw_people`), use [load_faces](utils.py) function for that.
2. Split the dataset into training and test sets.
3. Visualize some examples from training set with their corresponding label names.
4. Normalize the feature values by scaling them.

**Task 2: Determine the optimal number of components**. **Points: 20**

This step helps you understand what minimum number of components (eigenfaces) will be good to be used for model
training.

1. Perform PCA on the training data without reducing the number of components.
2. Calculate the explained variance ratios and their cumulative sum (use `explained_variance_` of PCA and `np.cumsum`).
3. Plot explained variance cumulative sum graph.
4. Determine the optimal number of components to retain by selecting the smallest number of components that explain at
   least 95% of the total variance.
5. Reconstruct faces using only optimal number of components and plot several examples, comparing original images
   with reconstructed ones.

**Task 3: Train a classifier**. **Points: 15**

1. Perform PCA on the training data with the optimal number of components.
2. Train a classifier of your choice using the PCA-transformed data. You can use any classifier, but make sure to use
   the same classifier with the same hyperparameters for all experiments to ensure a fair comparison.
3. Evaluate the classifier on the test set and report the chosen classification metric(s) and explain why you chose that
   particular metric(s).
4. Report training and mean inference time.

**Task 4: Compare classifier performance with different numbers of components**. **Points 20**

1. Repeat steps 1-4 of Task 3 for a smaller and a larger number of components than the optimal value determined in Task 2.
2. Repeat steps 1-4 of Task 3 using the full set of features (i.e., without PCA).

**Task 5: Analyze the results**. **Points 15**

1. Compare the classifier performance, training time, and inference time for the different numbers of components and the
   full set of features. Comment on overfitting. 
2. Discuss the trade-offs between component number, model performance, and computation time.
3. Provide a conclusion on the optimal number of components for this classification task based on your analysis.

**Code Quality and Report Quality**. **Points 20**

Insert your utility functions in [utils](utils.py) and your final report in the [Report notebook](Report.ipynb)

Please **note** that in this assignment you are not required to use separate validation set, you can do all your 
intermediate comparisons and reports on test set. 
