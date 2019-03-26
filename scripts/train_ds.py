# -----------------------------------------------------
# Authors: KrishnaSandeep Gaddam, Chandana Dasari
# Description: script to build a classification model using
# Decision Tree using mturk_sample.csv and Davis and Skene algorithm
# -----------------------------------------------------

# importing necessary packages
from collections import Counter
from sklearn import tree
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# Initialise the polarities by using maxium voting

# Read the mturk_sample dataset
mturk_data = pd.read_csv('../data/mturk_sample.csv')
mturk_temp = mturk_data.copy()

# Converted the labels to numeric values, 1 for 'pos' label and -1 for 'neg' label
mturk_temp['class'] = mturk_temp['class'].map({'pos': 1, 'neg': -1})

# Calculate mean of each column. By using mean,
# the feature values belonging to the same movie review id remains same,
# whereas the label mean results in a number in the range of -1 to 1.
mturk_data_ds = mturk_temp.groupby(['id']).mean()

# Converted to positive (1) or negative (0) labels using a threshold value of 0.
mturk_data_ds['class'] = mturk_data_ds['class'].apply(lambda x: 1 if x >= 0 else 0)

# Populate the polarities
polarities = dict()
for i, row in mturk_data_ds.iterrows():
    polarities[i] = {'pos': row['class'], 'neg': 1-row['class']}

# Get the list of unique workers
workers = set(mturk_data['annotator'])

# Run the Davis and Skene for 20 iterations
for e in range(20):

    # Initialise Worker confusion matrix
    wa = {w: {'tp': 0.0, 'fn': 0.0, 'fp': 0.0, 'tn': 0.0} for w in workers}

    # Update Confusion matrix of each worker based on polarities
    for i, row in mturk_data.iterrows():
        if row['class'] == 'pos':
            if polarities[row['id']]['pos'] > polarities[row['id']]['neg']:
                wa[row['annotator']]['tp'] += polarities[row['id']]['pos']
            else:
                wa[row['annotator']]['fp'] += polarities[row['id']]['neg']
        else:
            if polarities[row['id']]['pos'] > polarities[row['id']]['neg']:
                wa[row['annotator']]['fn'] += polarities[row['id']]['pos']
            else:
                wa[row['annotator']]['tn'] += polarities[row['id']]['neg']

    # Normalise the confusion matrix for each worker
    for w in wa:
        sum_tp_fn = (wa[w]['tp'] + wa[w]['fn'])
        sum_tn_fp = (wa[w]['tn'] + wa[w]['fp'])
        if sum_tp_fn > 0:
            wa[w]['tp'] = wa[w]['tp'] / sum_tp_fn
            wa[w]['fn'] = wa[w]['fn'] / sum_tp_fn
        if sum_tn_fp > 0:
            wa[w]['fp'] = wa[w]['fp'] / sum_tn_fp
            wa[w]['tn'] = wa[w]['tn'] / sum_tn_fp

    # Reset polarities
    polarities = {p: {'pos': 0.0, 'neg': 0.0} for p in polarities}

    # Update polarities using the updated confusion matrices of workers
    for i, row in mturk_data.iterrows():
        if row['class'] == 'pos':
            polarities[row['id']]['pos'] += wa[row['annotator']]['tp']
            polarities[row['id']]['neg'] += wa[row['annotator']]['fp']
        else:
            polarities[row['id']]['pos'] += wa[row['annotator']]['fn']
            polarities[row['id']]['neg'] += wa[row['annotator']]['tn']

    # Normalise polarities
    for p in polarities:
        sum_polarities = polarities[p]['pos'] + polarities[p]['neg']
        polarities[p]['pos'] /= sum_polarities
        polarities[p]['neg'] /= sum_polarities

# Update the final polarities back to the mturk_data_ds dataframe
for p in polarities:
    mturk_data_ds.at[p, 'class'] = int(polarities[p]['pos'] > polarities[p]['neg'])

# Preparing the train and test data
test_data = pd.read_csv('../data/test.csv')
class_mapping = {'pos': 1, 'neg': 0}

X_train = mturk_data_ds.drop(columns=['class'])
y_train = mturk_data_ds['class']

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

print("Model trained with Dawid & Skene method (crowd source)")
print("-------------------------------------")
print("Accuracy of the model   : {:0.2f}".format(accuracy*100))
print("F1 Score                : {:0.2f}".format(f1score*100))

# saving the results to the train_ds.txt in the result folder
with open('../results/train_ds.txt', 'w') as f:
    f.write("Model trained with Dawid & Skene method (crowd source)\n")
    f.write("-------------------------------------\n")
    f.write("Accuracy of the model   : {:0.2f}\n".format(accuracy * 100))
    f.write("F1 Score                : {:0.2f}\n".format(f1score * 100))
    f.write("Probabilities : \n")
    np.savetxt(f, y_predict_proba, fmt="%0.3f")
