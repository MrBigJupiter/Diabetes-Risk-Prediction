# Diabetes Risk Classification

This notebook presents a binary classification analysis using the diabetes dataset from `sklearn.datasets`. The objective is to determine whether a patient is **Endangered** or **Not Endangered** based on their target value under different risk thresholds.

## Dataset

The dataset is loaded using `load_diabetes(scaled=False)` and consists of multiple numerical features describing patient health indicators, along with a continuous target variable representing disease progression.

## Data Exploration

The dataset is explored to understand:
- Feature distributions and summary statistics
- The range and distribution of the target variable
- Potential class imbalance introduced by different threshold values

## Classification Scenarios

Two binary classification scenarios are defined using the target variable:

- **Scenario 1:**  
  Target values greater than **150** are classified as *Endangered*; all others as *Not Endangered*.

- **Scenario 2:**  
  Target values greater than **250** are classified as *Endangered*; all others as *Not Endangered*.

Each scenario is treated as a separate classification problem.

## Visualization

Visualizations are used to:
- Analyze the distribution of the target variable
- Observe class separation under different thresholds
- Evaluate model performance and prediction behavior
