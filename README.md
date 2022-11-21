# Santander Customer Transaction Prediction

## Content

### 1. Project Indroduction

#### 1.1 Background

#### 1.2 Problem Statement

### 2. Technical Specifications

#### 2.1 Software & Tools

#### 2.2 Necessary Files

### 3. Analysis

#### 3.1 Methodology

#### 3.2 Data Exploration

3.2.1 Class Imbalance

3.2.2 Correlated Features and Distributions

3.2.3 Fake Data Identification

#### 3.3 Modeling

3.3.1 Feature Engineering - Magic Features

3.3.2 Feature Engineering - Data upsampling

3.3.3 Model Comparison

3.3.4 Models Performance Evaluation

3.3.5 Why LightGBM

#### 3.4 Best Performing Model

3.4.1 Bayesian Optimization to find best parameters

3.4.2 Stratified K Fold with Magic Features

3.4.3 Feature Importance

### 4. Results and Future Steps

#### 4.1 Results & Insights

#### 4.2 Future Steps

### 5. Appendix

#### 5.1 Final Predictions

### 1.1 Background

Banco Santander is a multinational financial services company based in Santander, Spain. Santander’s US subsidiary, Santander Bank, focuses on the Northeast of the US - with 650 retail banks and over 2,000 ATMs in the region. Santander’s mission is to help people and businesses prosper - they are looking for ways to help customers understand their financial health.

### 1.2 Problem Statement

This has led their Data Science team to the following problem: Is there a way to determine which customers will make a specific transaction in the future, irrespective of the amount of money transacted? For this problem, Santander provided a dataset on Kaggle.

### 2.2 Necessary Files

train.csv - the training set.
test.csv - the test set. The test set contains some rows which are not included in scoring.
sample_submission.csv - a sample submission file in the correct format.

### 3.1 3.1. Methodology

After understanding the business background and the problem we seek to solve, we need to explore the dataset. There are 200k observations and 200 features in the training dataset and same size for the testing data. Considering all these 200 features we have, we need to find the most relevant features that help to make better predictions. The next step is feature engineering since the features that go into the model are crucial for training a good prediction model. We then checked for the missing values and repeated values and found that there is none of them.

Then, we started to try different models including Logistic Regression, Random Forest, XGBoost and LightGBM. We compared these models and decided to stick with LightGBM. Finally after all algorithms have been validated, trained and tested, we ended up using LightGBM model with the help of Bayesian Optimization to pick the best hyperparameters and Stratified K Fold to elevate the AUC score on our validation dataset. After the model being built, by plotting the feature importance among all the features we have, we know the weighted contribution of each feature to our final predictions.
