# -----------------------------------------------------
# Authors: KrishnaSandeep Gaddam, Chandana Dasari
# Description: script to build a classification model using Decision Tree using gold_sample.csv
# -----------------------------------------------------

# importing necessary packages
from sklearn import tree
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# Loading train and test data
train_data = pd.read_csv('../data/gold_sample.csv')
test_data = pd.read_csv('../data/test.csv')
# Mapping the class labels to 1(pos) , 0(neg)
class_mapping = {'pos': 1, 'neg': 0}

# Preparing the train data
X_train = train_data.drop(columns=['id', 'class'])
y_train = train_data['class']
y_train = y_train.map(class_mapping)

# preparing the test data
X_test = test_data.drop(columns=['id', 'class'])
y_test = test_data['class']
y_test = y_test.map(class_mapping)

# Decison tree classifier using the sklearn library
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

# Predicting the test labels
y_predict = clf.predict(X_test)
# Test Prediction probbilities
y_predict_proba = clf.predict_proba(X_test)

# calculating performance metrics
accuracy = clf.score(X_test, y_test)
f1score = f1_score(y_test.values, y_predict)


print("Model trained with Gold standard data")
print("-------------------------------------")
print("Accuracy of the model   : {:0.2f}".format(accuracy*100))
print("F1 Score                : {:0.2f}".format(f1score*100))

# saving the results to the train_gold.txt in the result folder
with open('../results/train_gold.txt', 'w') as f:
    f.write("Model trained with Gold standard data\n")
    f.write("-------------------------------------\n")
    f.write("Accuracy of the model   : {:0.2f}\n".format(accuracy * 100))
    f.write("F1 Score                : {:0.2f}\n".format(f1score * 100))
    f.write("Probabilities : \n")
    np.savetxt(f, y_predict_proba, fmt="%0.3f")
