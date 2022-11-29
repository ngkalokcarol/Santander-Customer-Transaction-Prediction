# Santander Customer Transaction Prediction
### Garett Carlblom, Noah Corwin, Yan Dong, Carol Ng, Ziyan Wu

# Content
## 1. Project Indroduction
>1.1 Background \
>1.2 Problem Statement
## 2. Technical Specifications
>2.1 Software & Tools \
>2.2 Necessary Files
## 3. Analysis
>3.1 Methodology \
>3.2 Data Exploration
>>3.2.1 Class Imbalance \
>>3.2.2 Correlated Features and Distributions \
>>3.2.3 Fake Data Identification \

>3.3 Modeling
>>3.3.1 Feature Engineering - Magic Features \
>>3.3.2 Feature Engineering - Data upsampling \
>>3.3.3 Model Comparison \
>>3.3.4 Models Performance Evaluation \
>>3.3.5 Why LightGBM \

>3.4 Best Performing Model \
>>3.4.1 Bayesian Optimization to find best parameters \
>>3.4.2 Stratified K Fold with Magic Features \
>>3.4.3 Feature Importance
## 4. Results and Future Steps
> 4.1 Results & Insights \
> 4.2 Future Steps \

## 5. Appendix
> 5.1 Final Predictions \



## 1. Project Indroduction

### 1.1 Background

Banco Santander is a multinational financial services company based in Santander, Spain. Santander’s US subsidiary, Santander Bank, focuses on the Northeast of the US - with 650 retail banks and over 2,000 ATMs in the region. Santander’s mission is to help people and businesses prosper - they are looking for ways to help customers understand their financial health.

### 1.2 Problem Statement


This has led their Data Science team to the following problem: Is there a way to determine which customers will make a specific transaction in the future, irrespective of the amount of money transacted? For this problem, Santander provided a dataset on Kaggle.

## 2. Technical Specifications
### 2.1 Software & Tools

 
Python Version 3.7.13    | 
-------------------|
Numpy              | 
Pandas             | 
sklearn            |
matplotlib              | 
xgboost             | 
tensorflow            |
lightgbm            |
scipy              | 
skopt             | 
bayes_opt            |



### 2.2 Necessary Files
>* train.csv - the training set.
>*test.csv - the test set. The test set contains some rows which are not included in scoring.
>*sample_submission.csv - a sample submission file in the correct format.


## 3. Analysis

### 3.1. Methodology
After understanding the business background and the problem we seek to solve, we need to explore the dataset. There are 200k observations and 200 features in the training dataset and same size for the testing data. Considering all these 200 features we have, we need to find the most relevant features that help to make better predictions. The next step is feature engineering since the features that go into the model are crucial for training a good prediction model. We then checked for the missing values and repeated values and found that there is none of them. 

Then, we started to try different models including Logistic Regression, Random Forest, XGBoost and LightGBM. We compared these models and decided to stick with LightGBM. Finally after all algorithms have been validated, trained and tested, we ended up using LightGBM model with the help of Bayesian Optimization to pick the best hyperparameters and Stratified K Fold to elevate the AUC score on our validation dataset. After the model being built, by plotting the feature importance among all the features we have, we know the weighted contribution of each feature to our final predictions.

![image](https://user-images.githubusercontent.com/50436546/204637646-8753d75f-c83e-420f-b97a-274d5ae8601a.png)


### 3.2. Data Exploration


#### 3.2.1 Class Imbalance

Before we make predictive models, the first thing we need to do is data exploration and visualization. Firstly, we want to check the class balance and we found that the training dataset is pretty imbalanced, with 90% of the customers not making transactions and only 10% making transactions. We will consider using techniques such as downsampling, upsampling, or adjusting the class weights when training models.


```python
train.groupby('target').size().plot(kind='pie',y = "target",label = "Type",autopct='%1.1f%%')
```

![image](https://user-images.githubusercontent.com/50436546/204376508-5d9eb097-23a3-413e-acae-c1330dcccd49.png)


#### 3.2.2 Correlated Features and Distributions

Then, we checked the missing data and to make sure there is no missing data. Considering there are 200 features, we want to know if we can remove the highly correlated features. 

We plot the correlations between features and found out that there is no strong correlation between features, so we cannot directly eliminate the correlated features but need to use all the features.

![image](https://user-images.githubusercontent.com/50436546/204376599-f8db4bdb-a771-4207-9ecf-b34f40130792.png)

Usually the features that is highly correlated with the target are more important than other features. We want to explore the correlation between each feature with our target, whether or not the customer make a specific transaction in the future. Again, there is no strong correlation, so we just going to use all the features at first.

![image](https://user-images.githubusercontent.com/50436546/204376660-f611e8ce-441a-40c6-a8a5-076d0782d516.png)

```python
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/50436546/204636403-74646849-8047-4306-82f8-e5e42fdbc312.png)


We also want to know if the training data and testing data are similar. If there is any fundamental difference, the model we train on training data cannot perform well on testing data.

![image](https://user-images.githubusercontent.com/50436546/204376723-450e600a-7baf-4504-92f5-f4e3a4f40e5c.png)

![image](https://user-images.githubusercontent.com/50436546/204376738-f381f8e7-4716-42df-8c6e-0051350361df.png)

There is almost no difference between the standard deviations of training and testing data. Now we take a look at the means.

![image](https://user-images.githubusercontent.com/50436546/204376873-7428c30e-b099-4cbe-9b84-d4bac0d0d11d.png)

![image](https://user-images.githubusercontent.com/50436546/204376893-7eef1bd3-eca4-4df5-a9ef-891d341c5411.png)

Again, there is no big difference between the means of each feature of training and testing data. Only few features show slight differences.

We also looked into duplicated values from the train and test data set, we found that there are duplicated values with the same or extremely close values, of which the pattern can be used for further feature engineering. 


```python
features = train.columns.values[2:202]
unique_max_train = []
unique_max_test = []

for feature in features:
    values = train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])
```


```python
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(15))
```
![image](https://user-images.githubusercontent.com/50436546/204376948-fb3c920b-0e04-4763-a00b-f58d44e0adb4.png)

![image](https://user-images.githubusercontent.com/50436546/204376965-8a3bd5a5-341e-4c70-a763-d2718bf9acde.png)


#### 3.2.3 Fake Data Identification

According to many discussions in Kaggle, there are some synthetic samples in the test data.

https://www.kaggle.com/code/yag320/list-of-fake-samples-and-public-private-lb-split

https://github.com/btrotta/kaggle-santander-2019

We plot the the distribution of the number of unique values in each column and find out that the training data have much more unique values than the testing data. This could be that there are some synthetic data in the testing dataset.

![image](https://user-images.githubusercontent.com/50436546/204377071-a8fcba7e-1652-4bb7-a812-3f2cbf970ec0.png)

If none of the values in a row is unique, we identify this row as synthetic data. In this way we can find out the fake data in the testing dataset.


```python
# identify the fake test data
feature_names = train.columns.values.tolist()[2:]
unique_value = {}

for i in feature_names:
    sizes = test.groupby(i)['ID_code'].size()
    unique_value[i] = sizes.loc[sizes == 1].index
unique_bool = test.copy()

for i in feature_names:
    unique_bool[i] = test[i].isin(unique_value[i])
num_unique = unique_bool[feature_names].sum(axis=1)
test_real = test.loc[num_unique > 0]
```

There are 100000 real samples in testing data and 100000 synthetic samples.

###3.3 Modeling

#### 3.3.1 Feature Engineering - Magic Features

If some values repeat more times than other values, this means these signals are more important. So we created two magic features for each column, one is the size of the same values in the whole real dataset, another one is average size of the same values within its bin. Here we divided each column into 300 bins. These 400 new features can amplify the signals of the repeated values.


```python
# add the counts of unique value and scaled counts of unique value
all_data = pd.concat([train, test], ignore_index=True, sort=True).reset_index()
count_data = pd.concat([train, test_real], axis=0, sort=True)
for c in train.columns.values.tolist()[2:]:
    all_size = count_data.groupby(c)['ID_code'].size().to_frame(c + '_size')
    count_data['rank'] = count_data[c].rank(method='first')
    count_data['bin'] = (count_data['rank'] - 1) // 300
    count_data['size'] = count_data.groupby(c)['ID_code'].transform('size')
    count_data['avg_size_in_bin'] = count_data.groupby('bin')['size'].transform('mean')
    count_data[c + '_size_scaled'] = count_data['size'] / count_data['avg_size_in_bin']
    all_size[c + '_size_scaled'] = count_data.groupby(c)[c + '_size_scaled'].first()
    all_data = pd.merge(all_data, all_size, 'left', left_on=c, right_index=True)

train = all_data.loc[all_data['target'].notnull()].copy()
test = all_data.loc[all_data['target'].isnull()].copy()
train.drop(['index'], axis=1, inplace=True)
test.drop(['index','target'], axis=1, inplace=True)
```

#### 3.3.2 Feature Engineering - Data upsampling

At the EDA phase mentioned earlier, there was a class imbalance of the dataset, so we tried to upsample the data by duplicating smaller classes to make the whole dataset more equal and similar, however, since the original distribution of the data is changed, our upsampling did not improve the model performance so we dropped this step.

Partial output

![image](https://user-images.githubusercontent.com/50436546/204377175-4d20bf36-8952-4c59-8ee1-6f3bbbb2db85.png)

### 3.3.3 Model Comparison

To help our team discover which is the best model for our team to use for the final score, we first ran 4 different models and compared their scores using nested cross validation. We ran Logistic Regression, Random Forest, XGBoost, and LightGBM models. The implementation of these models was very basic, after picking the best basic model we can then focus our attention on the model and tune it to be more effective at prediction. 


```python
# Comparing the model results of Nested cross validation
print('Logistic Regression AUC:', sum(lr_scores) / len(lr_scores))
print('Random Forest AUC:', sum(rfc_scores) / len(rfc_scores))
print('XGBoost AUC:', sum(boost_scores) / len(boost_scores))
print('LGBM AUC:', sum(lgbm_scores) / len(lgbm_scores))
```

    Logistic Regression AUC: 0.8504492188811504
    Random Forest AUC: 0.7772756710766303
    XGBoost AUC: 0.8483291494548304
    LGBM AUC: 0.8483291494548304


#### 3.3.4 Models Performance Evaluation


The performance metric we used to compare different models is the Area Under Curve(AUC) which represents the probability that a random positive sample is ranked higher than a random negative sample.



Model	Area Under Curve (AUC)
LightGBM	0.84832
Logistic Regression	0.85045
Random Forest	0.77728
XGBoost	0.84832


#### 3.3.5 Why LightGBM



Out of the four models, we had very similar scores for all of them except Random Forest. Our team ended up deciding on using LightGBM, as it has a wide variety or tuning options (compared to Logistic Regression) and is more efficient to run (compared to XGBoost).

### 3.4 Best Performing Model


#### 3.4.1 Bayesian Optimization to find best parameters

To find the best parameters to use in our LightGBM model, our team used Bayesian Optimizaiton. This strategy uses the Bayes Theorem of statistics to help find the optimized parameters to use. 

Implementation was found through the following link: https://www.kaggle.com/code/sz8416/simple-bayesian-optimization-for-lightgbm/notebook


```python
def bayes_parameter_opt_lgb(X, Y, init_round=15, opt_round=25, n_folds=1, random_seed=45, n_estimators=10000, 
                            learning_rate=0.05, scale_pos_weight=8.95, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=Y, categorical_feature = 'auto', free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, max_bin, min_child_samples, reg_alpha, reg_lambda, feature_fraction, bagging_fraction, 
                 bagging_freq, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 
                  'early_stopping_round':100, 'scale_pos_weight':scale_pos_weight, 'metric':'auc'}
        params['num_leaves'] = int(round(num_leaves))
        params['max_bin'] =  int(round(max_bin))
        params['min_child_samples'] = int(round(min_child_samples))
        params['reg_alpha'] =  int(round(reg_alpha))
        params['reg_lambda'] = int(round(reg_lambda))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['bagging_freq']= int(round(bagging_freq))
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, 
                           stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 33),
                         'max_bin': (1000, 1050), 
                         'min_child_samples': (900, 1100),
                         'reg_alpha': (.05, 2.0),
                         'reg_lambda': (0.1, 0.3),
                         'feature_fraction': (0.5, 1.5),
                         'bagging_fraction': (0.8, 1.2),
                         'bagging_freq': (1,  3),
                         'max_depth': (3, 8),
                         'lambda_l1': (0, 7),
                         'lambda_l2': (0, 5),
                         'min_split_gain': (0.001, 0.0),
                         'min_child_weight': (.001, .1)}, random_state=45)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.max   #['max']['max_params']

opt_params = bayes_parameter_opt_lgb(X, Y, init_round=5, opt_round=10, n_folds=3, random_seed=45, 
                  n_estimators=100, learning_rate=0.05, scale_pos_weight=8.95)
```


```python
# parameters learned from Bayesian Optimization
param = {
  'learning_rate': .05,
  'bagging_fraction': 0.902521322815825,
  'bagging_freq': 1,
  'feature_fraction': 1.0,
  'lambda_l1': 4.1721172821564645,
  'lambda_l2': 0.016372953387883693,
  'max_bin': 1050,
  'max_depth': 8,
  'min_child_samples': 1023,
  'min_child_weight': 0.051005148095158,
  'min_split_gain': 0.0,
  'num_leaves': 27,
  'reg_alpha': 0.05557302129673095,
  'reg_lambda': 0.2965997314178622, 'objective': 'binary', 'metric':'auc',  'n_estimators':100, 'learning_rate':0.05, 'boost_from_average':'false'}
```

#### 3.4.2 Stratified K Fold with Magic Features

In the final training of our model we used Stratified K Fold validation. In this technique the model is trained and validated using stratified samples of the data K times, with K being chosen by the user. Each iteration samples different portions of the input data to get the training and validation data, from this we get an AUC score for each iteration. Taking the average of these scores makes sure that randomness in the model training is minimized. A major advantage to using Stratified K Fold validation over other techniques such as K fold validation, is that the proportions of the imblanced data will be preserved in Stratified K Fold validation. 


To help with the stratified K fold implementaiton we looked to this post: https://www.kaggle.com/code/jesucristo/30-lines-starter-solution-fast


```python
folds = StratifiedKFold(n_splits=6, shuffle=False)
oof = np.zeros(len(train)) 
#used to show ROC score on validation tests, not needed for testing prediciton
predictions = np.zeros(len(test)) 
# mimics the testing data but with zero values, the predictions for each value will appended to this 
```


```python
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    train_dataset = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])  
    # sets up training data
    val_dataset = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])  
    # sets up validation data
    clf = lgb.train(param, train_dataset, 1000000, valid_sets = [train_dataset, val_dataset], verbose_eval=5000, early_stopping_rounds = 2000) 
    # trains the model
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) 
    # used to calculate the roc score in the following cell, not needed in the testing prediction
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits  
    # the division of folds.n_splits is a great way to get the average of the best predictions for each k fold
    # use best_iteration to get the best iteration when using early stopping
```


```python
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
```

    CV score: 0.90885 


#### 3.4.3 Feature Importance
Feature importance assigns the score of input features based on their importance to predict the taget. In this case, the feature importance show how all of these features contribute to predicting if the customer will make this specific transaction in the future. The more the features will be responsible to predict the outcome, the higher will be their scores.

![image](https://user-images.githubusercontent.com/50436546/204635915-55162c88-4789-4b7a-a0fc-175a200bc8cc.png)


This graph above shows the features that have the 10 highest feature importance scores. The score itself does not necessarily have business value, but we can know from this graph that some features contribute more than others. We have the top 5 features, var_174, var_110, var_53, var_22, and var_146. The description of these features is not provided with us, so we can only conclude that these are the most important features of a customer that influence their transaction behavior.

## 4. Results and Future Steps

### 4.1 Results & Insights
>Finally after all algorithms have been validated, trained and tested, we ended up using LightGBM model which achieved an AUC score of 0.90885 on the validation dataset and 0.91021 on the test dataset with the help of Bayesian Optimization to pick the best hyperparameters and Stratified K Fold to elevate the AUC score on our validation dataset. After the model being built, by plotting the feature importance among all the features we have, we allocated the weighted contribution of each feature to our final predictions. Although we don't have any background knowledge what these features mean, Santander would likely find those valuable on their end.

### 4.2 Future Steps
* We believe our model could be further improved via more nuanced approaches to the inclusion of magic features in the datasets.

* Stacking our LightGBM Model with a Neural Network or another LightGBM Model may also improve performance.




## 5. Appendix
### 5.1 Final Predictions

![image](https://user-images.githubusercontent.com/50436546/204377451-0603b46f-4517-4920-a195-afe6a8b0a3fe.png)

### 5.2 Reference

https://www.kaggle.com/code/gpreda/santander-eda-and-prediction 
https://www.kaggle.com/code/allunia/santander-customer-transaction-eda
https://www.kaggle.com/code/mjbahmani/santander-ml-explainability
