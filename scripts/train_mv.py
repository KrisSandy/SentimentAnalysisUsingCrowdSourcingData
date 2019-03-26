# -----------------------------------------------------
# Authors: KrishnaSandeep Gaddam, Chandana Dasari
# Description: script to build a classification model using
# Decision Tree using mturk_sample.csv and miximum voting
# -----------------------------------------------------

# importing necessary packages
from sklearn import tree
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# Read the mturk_sample dataset
mturk_data = pd.read_csv('../data/mturk_sample.csv')

# Converted the labels to numeric values, 1 for 'pos' label and -1 for 'neg' label
mturk_data['class'] = mturk_data['class'].map({'pos': 1, 'neg': -1})

# Calculate mean of each column. By using mean,
# the feature values belonging to the same movie review id remains same,
# whereas the label mean results in a number in the range of -1 to 1.
mturk_data_mv = mturk_data.groupby(['id']).mean()

# Converted to positive (1) or negative (0) labels using a threshold value of 0.
mturk_data_mv['class'] = mturk_data_mv['class'].apply(lambda x: 1 if x >= 0 else 0)

# Preparing the train and test data
test_data = pd.read_csv('../data/test.csv')
class_mapping = {'pos': 1, 'neg': 0}

X_train = mturk_data_mv.drop(columns=['class'])
y_train = mturk_data_mv['class']

X_test = test_data.drop(columns=['id', 'class'])
y_test = test_data['class']
y_test = y_test.map(class_mapping)

# Decison tree classifier using the sklearn library
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

# Predicting the test labels
y_predict = clf.predict(X_test)
y_predict_proba = clf.predict_proba(X_test)

# calculating performance metrics
accuracy = clf.score(X_test, y_test)
f1score = f1_score(y_test.values, y_predict)

print("Model trained with Majority Vote data (crowd source)")
print("-------------------------------------")
print("Accuracy of the model   : {:0.2f}".format(accuracy*100))
print("F1 Score                : {:0.2f}".format(f1score*100))

# saving the results to the train_mv.txt in the result folder
with open('../results/train_mv.txt', 'w') as f:
    f.write("Model trained with Majority Vote data (crowd source)\n")
    f.write("-------------------------------------\n")
    f.write("Accuracy of the model   : {:0.2f}\n".format(accuracy * 100))
    f.write("F1 Score                : {:0.2f}\n".format(f1score * 100))
    f.write("Probabilities : \n")
    np.savetxt(f, y_predict_proba, fmt="%0.3f")
