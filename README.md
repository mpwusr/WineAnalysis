# Wine Quality Dataset Analysis

This repository contains a Python script that performs an exploratory data analysis and modeling of the UCI Wine Quality Dataset (red wine variant). The analysis includes data summary, visualization, regression, and classification techniques.

## Dataset
- Source: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- File: `winequality-red.csv`
- Contains 1599 samples with 11 physicochemical features and a quality score

## Features
1. **Data Exploration**
   - Dataset summary statistics
   - Visualization of quality distribution (bar plot)
   - Histogram of fixed acidity
   - Basic statistics (mean, median, mode) of quality scores

2. **Regression Analysis**
   - Simple Linear Regression (using fixed acidity to predict quality)
   - Multiple Linear Regression (using all features to predict quality)
   - Performance metrics (MSE, R² scores)
   - Visualization of regression results

3. **Classification Analysis**
   - Decision Tree Classifier
   - Softmax Regression (Multinomial Logistic Regression)
   - Model evaluation with training/testing accuracy
   - Example predictions with probability estimates
   - Confusion matrix visualization

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Optional for decision tree visualization:
# pip install graphviz pydotplus
# conda install python-graphviz (if using conda)
