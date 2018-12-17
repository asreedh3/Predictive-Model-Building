# Predictive-Model-Building
Building a Predictive Model from Scratch Using Python Libraries
The dataset used for this particular task has the following specifications
Rows:2500
Attributes/Features:66
Feature Typs: Mix (Categorical and Continuous)
Problem Type: Classification (Binary)
Class Distribution: Unbalanced

Model Building Methodology:

1) Data Preprocessing: One Hot Encoding for all Categorical Variables.

2) Sampling Procedures: Creation of Independent Test Set (No Duplication of Instances. Upsampling done on train set for minority class.

3) Base Model Selection for Hyperparameter Tuning: Selection of the best machine learning model for hyperparameter exploration. Alternatives considered include: SVM, Random Forest and AdaBoost.

4) Feature Selection Using Best Base Model: Feature Selection done using sklearn's in-built SelectFromModel functionality.

5) Randomized Grid Search for Hyperparameters: Randomized Grid Search done to narrow the Grid Space for HYperparameter Tuning.

6) Grid Search for Hyperparameters: Explicit Grid Search run with Narrowed Grid Space obtained from Randomized Grid Search. All explicit Combinations of Hyperparameters explored.

7) Final Model Selection: Final Model Performance Evaluated based on criteria such as ROC Curves, Balanced Error Rate, Matthews Correlation Coefficient and F1-Score.
