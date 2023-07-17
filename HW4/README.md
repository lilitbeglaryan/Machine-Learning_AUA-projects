[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10486552&assignment_repo_type=AssignmentRepo)
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

[knn](knn.py) **POINTS 100**

In this assignment, you will implement a KNN classifier and evaluate its performance on a synthetic dataset with a large
number of features and a small number of samples. You will also experiment with different distance metrics and feature
normalizations.

Do your experiments in the dedicated KNN notebook.
You can se other modules from imported libraries, use can also use pandas, seaborn and any other lib that you find
necessary.

1. Generate a synthetic dataset with a large number of features (e.g., 1000) and a small number of samples (e.g., 100)
   using the `generate_data` function. **POINTS: 0**

2. Implement a KNN classifier as a class with an API similar to Sklearn's implementation. Fill the not implemented
   functions to train the KNN classifier on the synthetic dataset. **POINTS: 20**

3. Evaluate the performance of the KNN classifier using k-fold cross-validation. Record the accuracy of the classifier.
   Additionally, plot the learning curve of the KNN classifier for different values of k using the
   `evaluate_knn_performance` function and `the plot_learning_curve` function. **POINTS: 10**

4. Normalize the features of the dataset using both L1 and L2 normalization. **POINTS: 10**

5. Use different distance metrics such as Manhattan distance, Chebyshev distance, and Mahalanobis distance to train the
   KNN classifier on the original dataset.

6. Repeat step 2 and step 3 using the normalized dataset with different distances. Compare the performance of the KNN
   classifier with and without normalization for different distances. Additionally, plot the learning curve of the KNN
   classifier for different values of k, distances and normalization using the `evaluate_knn_performance` and
   `plot_learning_curve` functions. **POINTS: 20**

7. Find the best model (by distance, normalization, k) by validation metrics. Retrain on the whole train data and report
   its performance on initially held-out test data. **POINTS: 10**

8. Train and test Sklearn's implementation of KNN with your best KNNs hyperparameters and normalization technique (
   distance, normalization, k). Evaluate model on test data and compare with your best model. **POINTS: 10**

9. Your code quality, readability, report notebook structure adn readability will be checked and graded too. *POINTS:
   20*

