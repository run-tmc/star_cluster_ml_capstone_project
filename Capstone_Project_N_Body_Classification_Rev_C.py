# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:45:30 2017

@author: Trevor M. Clark
"""
"""Udacity NanoDegree Machine Learning Capstone Project"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


"""Import N_Body Dataset"""

n_body_star_data =\
 pd.read_csv("N_Body_Star_Classification_Capstone_Dataset_Rev_A.csv")

"""Dataset Preprocessing"""

"""Drop mass(m) label and data column from the dataframe since 
it is a constant value"""

n_body_star_data = n_body_star_data.drop('m', axis=1)

"""Drop star identification (id) label and data column from the 
dataframe since it is not a star feature for model classification"""

n_body_star_data = n_body_star_data.drop('id', axis=1)


"""Feature coefficient of determination (R^2) with a DecisionTreeRegressor"""
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn import tree

# Create dataframe with star classifications
target_classifications = n_body_star_data['Star_Classification']
 
# Create dataframe without the target 'star_classification' classifer label
feature_data = n_body_star_data.drop('Star_Classification', axis=1)

# Create list of the r2_data feature labels
feature_data_col_list = feature_data.axes[1]

# Establish for loop to compute R^2 coefficients of determination 
# for each feature with a decision tree regressor 

feature_regressor = tree.DecisionTreeRegressor(random_state=0)
for removed_feature in feature_data_col_list:
    # Generate a dataform of the classification feature for R^2 analysis 
    target_feature = pd.DataFrame(feature_data[removed_feature])
    # Generate a dataform without the classification feature
    feature_data_subset =\
    pd.DataFrame(feature_data.drop([removed_feature],axis=1))
    
    X_train, X_test, y_train, y_test =\
    cross_validation.train_test_split\
    (feature_data_subset,target_feature,test_size=0.25, random_state=0)
    
    feature_regressor = feature_regressor.fit(X_train, y_train)
    y_pred = feature_regressor.predict(X_test)
    score = round(r2_score(y_test, y_pred),3)
    print "The R^2 coefficient of determination is",\
    score, "for feature", removed_feature

"""Import SVM, and Logistic Regression classifiers"""
from sklearn import svm
from sklearn import linear_model

"""Initialize the classifer models"""
bnchmk_d_tree_classifier = tree.DecisionTreeClassifier(random_state=0)
support_vector_classifier = svm.SVC(random_state=0)
log_regression_classifier = linear_model.LogisticRegression(random_state=0)

"""Split dataset to 40%/60%, 60%/40%, and 70%/30% training test set sizes"""

X_train_40, X_test_40, y_train_40, y_test_40 =\
cross_validation.train_test_split\
(feature_data, target_classifications, test_size = 0.6, random_state = 0)

X_train_60, X_test_60, y_train_60, y_test_60 =\
cross_validation.train_test_split\
(feature_data, target_classifications, test_size = 0.4, random_state = 0)

X_train_70, X_test_70, y_train_70, y_test_70 =\
cross_validation.train_test_split\
(feature_data, target_classifications, test_size = 0.3, random_state = 0)


"""Define function to train and test classifier; and 
evaluate and display classifer performance"""
def classifer_train_test_evaluation(clf, X_train, y_train, X_test, y_test):
    # Train classifer with training set features and classifications
    clf.fit(X_train, y_train)
     # Use train features to predict classifications with trained algorithm
    y_train_pred = clf.predict(X_train)
    # Use test features to predict classifications with trained algorithm
    y_test_pred = clf.predict(X_test)
    # Evaluate classifer prediction performance with training features
    train_performance = accuracy_score(y_train, y_train_pred)
     # Evaluate classifer prediction performance with test features
    test_performance_accuracy = accuracy_score(y_test, y_test_pred)
    # Evaluate f1 score for each target classification
    # Create list of target cluster classifications
    f1_position_label_list = ['In_Cluster','Isolated']
    # Initialize f1 score list
    test_performance_f1_list = [0]*2
    # Determine the f1 score for each target classification    
    for i in range(0,len(f1_position_label_list)):
        test_performance_f1_list[i] = f1_score\
        (y_test, y_test_pred, pos_label=f1_position_label_list[i])
    # Display the test performance results         
    print "Trained Classifer Algorithm: ",  clf.__class__.__name__
    print "Train Dataset Size: ", len(X_train)
    print "Training Accuracy Score: ", round(train_performance,7)
    print "Test Accuracy Score: ", round(test_performance_accuracy,7)
    for i in range(0,len(f1_position_label_list)):
        print "Test F1 Score for the {} classification is {}".format\
        (f1_position_label_list[i],round(test_performance_f1_list[i],7))

"""Train, test, and Evaluation the Benchmark Decision Tree Classifier 
Algorithm and Alternative Classifier Algorithms"""

# Establish list of classifiers
alt_classifier_list = [bnchmk_d_tree_classifier, support_vector_classifier,\
log_regression_classifier]

train_test_lists = [[X_train_40, y_train_40, X_test_40, y_test_40],\
[X_train_60, y_train_60, X_test_60, y_test_60],\
[X_train_70, y_train_70, X_test_70, y_test_70]]

# Train and cross validate classifiers with training/test set 
for alt_classifier in alt_classifier_list:
    for train_test_data in train_test_lists:
        classifer_train_test_evaluation\
        (alt_classifier, train_test_data[0], train_test_data[1],\
        train_test_data[2], train_test_data[3])
    
"""Import N_Body Population Dataset"""

n_body_star_population_data = pd.read_csv\
("N_Body_Star_Classification_Pre_Processing.csv")


print "Population Data"

"""Population Dataset Preprocessing"""

"""Drop mass(m) label and data column from the dataframe since
 it is a constant value"""

n_body_star_population_data = n_body_star_population_data.drop('m', axis=1)

"""Drop star identification (id) label and data column from the dataframe
since it is not a star feature for model classification"""

n_body_star_population_data = n_body_star_population_data.drop('id', axis=1)

# Create dataframe with star classifications
target_population_classifications =\
n_body_star_population_data['Star_Classification']
 
# Create dataframe without the target 'star_classification' classifer label
feature_population_data =\
n_body_star_population_data.drop('Star_Classification', axis=1)

"""Train Bench Mark and Support Vector Machine Algorithms with the sample 
dataset and Evaluate Model with N_Body Star Cluster Population Dataset"""

# Establish list of alternative classifiers
evaluation_classifier_list =\
 [bnchmk_d_tree_classifier, support_vector_classifier]

for alt_classifier in evaluation_classifier_list:
    classifer_train_test_evaluation(alt_classifier, X_train_70, y_train_70,\
    feature_population_data, target_population_classifications)




