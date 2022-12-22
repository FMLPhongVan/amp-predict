import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import Pfeature.pfeature as pf
from decision_tree import DecisionTree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

from random_forest import RandomForest


def print_hi(name):	
	# Use a breakpoint in the code line below to debug your script.
	print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def aac(input):
	a = input.rstrip('txt')
	output = a + 'aac.csv'
	df_out = pf.aac_wp(input, output)
	df_in = pd.read_csv(output)
	return df_in


def dpc(input):
	a = input.rstrip('txt')
	output = a + 'dpc.csv'
	df_out = pf.dpc_wp(input, output)
	df_in = pd.read_csv(output)
	return df_in


def feature_calc(po, ne, feature_name):
	# Calculate feature
	po_feature = feature_name(po)
	ne_feature = feature_name(ne)
	# Create class labels
	po_class = pd.Series([1 for i in range(len(po_feature))])
	ne_class = pd.Series([0 for i in range(len(ne_feature))])
	# Combine po and ne
	po_ne_class = pd.concat([po_class, ne_class], axis=0)
	po_ne_class.name = 'class'
	po_ne_feature = pd.concat([po_feature, ne_feature], axis=0)
	# Combine feature and class
	df = pd.concat([po_ne_feature, po_ne_class], axis=1)
	return df


def show_confusion_matrix(y_true, y_pred):
	confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[False, True]).plot()
	plt.show()


train_directory = './train_set'
test_directory = './test_set'
pos_train = train_directory + '/train_po_cdhit.txt'
neg_train = train_directory + '/train_ne_cdhit.txt'
feature_train = feature_calc(pos_train, neg_train, aac)
X = feature_train.drop(['class'], axis=1)
Y = feature_train.iloc[:, -1].values.reshape(-1, 1)
# Y = feature_train['class'].copy()

X_train, X_test_sample, Y_train, Y_test_sample = train_test_split(X, Y, test_size=0.3, random_state=47, shuffle=True, stratify=Y)
X_test_sample = X_test_sample.values

pos_test = test_directory + '/test_po.txt'
neg_test = test_directory + '/test_ne.txt'
feature_test = feature_calc(pos_test, neg_test, aac)
X_test = feature_test.iloc[:, :-1].values
Y_test = feature_test.iloc[:, -1].values.reshape(-1, 1)

# create a new csv file in records folder with name records-<current date and time>.csv
file = open('./records/records-' + str(datetime.datetime.now()) + '.csv', 'x')
file_sk = open('./records/records-sklearn-' + str(datetime.datetime.now()) + '.csv', 'x')
pd.DataFrame(columns=['depth', 'Validation ACC', 'Test ACC', 'Validation MCC', 'Test MCC']).to_csv(file, index=False)
pd.DataFrame(columns=['depth', 'SK Validation ACC', 'SK Test ACC', 'SK Validation MCC', 'SK Test MCC']).to_csv(file_sk, index=False)

rf = open('./records/records-rf-' + str(datetime.datetime.now()) + '.csv', 'x')
pd.DataFrame(columns=['depth', 'RF Validation ACC', 'RF Test ACC', 'RF Validation MCC', 'RF Test MCC']).to_csv(rf, index=False)
rf_sk = open('./records/records-rf-sklearn-' + str(datetime.datetime.now()) + '.csv', 'x')
pd.DataFrame(columns=['depth', 'RF SK Validation ACC', 'RF SK Test ACC', 'RF SK Validation MCC', 'RF SK Test MCC']).to_csv(rf_sk, index=False)

mode = 'gini'

for depth in range(1, 21):
	classifier = DecisionTree(max_depth=depth, min_samples_split=2, mode=mode)
	forest = RandomForest(n_trees=20, max_depth=depth, min_samples_split=2, mode=mode)
	classifier.fit(X_train, Y_train)
	forest.fit(X_train, Y_train)
	tree_file_name = 'test_results/tree{}-{}.txt'.format(classifier.max_depth, mode)
	classifier.print_tree_to_file(tree_file_name)

	Y_pred = classifier.predict(X_test)
	Y_rf_pred = forest.predict(X_test)
	# print accuracy result to test_result
	accuracy = accuracy_score(Y_test, Y_pred)
	rf_acc = accuracy_score(Y_test, Y_rf_pred)
	mcc = matthews_corrcoef(Y_test, Y_pred)
	rf_mcc = matthews_corrcoef(Y_test, Y_rf_pred)

	sk_classifier = DecisionTreeClassifier(criterion=mode, max_depth=classifier.max_depth, min_samples_split=classifier.min_samples_split)
	sk_rf = RandomForestClassifier(criterion=mode, max_depth=classifier.max_depth, min_samples_split=classifier.min_samples_split, n_estimators=forest.n_trees)
	sk_classifier.fit(X_train, Y_train)
	sk_rf.fit(X_train, Y_train)
	sk_Y_pred = sk_classifier.predict(X_test)
	sk_rf_Y_pred = sk_rf.predict(X_test)
	sk_accuracy = accuracy_score(Y_test, sk_Y_pred)
	sk_rf_acc = accuracy_score(Y_test, sk_rf_Y_pred)
	sk_mcc = matthews_corrcoef(Y_test, sk_Y_pred)
	sk_rf_mcc = matthews_corrcoef(Y_test, sk_rf_Y_pred)

	Y_pred_sample = classifier.predict(X_test_sample)
	Y_rf_pred_sample = forest.predict(X_test_sample)
	accuracy_sample = accuracy_score(Y_test_sample, Y_pred_sample)
	rf_acc_sample = accuracy_score(Y_test_sample, Y_rf_pred_sample)
	mcc_sample = matthews_corrcoef(Y_test_sample, Y_pred_sample)
	rf_mcc_sample = matthews_corrcoef(Y_test_sample, Y_rf_pred_sample)

	show_confusion_matrix(Y_test_sample, Y_pred_sample)
	sk_accuracy_sample = accuracy_score(Y_test_sample, sk_classifier.predict(X_test_sample))
	sk_rf_acc_sample = accuracy_score(Y_test_sample, sk_rf.predict(X_test_sample))
	sk_mcc_sample = matthews_corrcoef(Y_test_sample, sk_classifier.predict(X_test_sample))
	sk_rf_mcc_sample = matthews_corrcoef(Y_test_sample, sk_rf.predict(X_test_sample))

	accuracy_file_name = 'test_results/result{}-{}.txt'.format(classifier.max_depth, mode)

	pd.DataFrame([[classifier.max_depth, accuracy_sample, accuracy, mcc_sample, mcc]]).to_csv(file, mode='a', header=False, index=False)
	pd.DataFrame([[classifier.max_depth, sk_accuracy_sample, sk_accuracy, sk_mcc_sample, sk_mcc]]).to_csv(file_sk, mode='a', header=False, index=False)

	pd.DataFrame([[classifier.max_depth, rf_acc_sample, rf_acc, rf_mcc_sample, rf_mcc]]).to_csv(rf, mode='a', header=False, index=False)
	pd.DataFrame([[classifier.max_depth, sk_rf_acc_sample, sk_rf_acc, sk_rf_mcc_sample, sk_rf_mcc]]).to_csv(rf_sk, mode='a', header=False, index=False)

	with open(accuracy_file_name, 'w') as f:
		f.write('Validation Accuracy: {}\n'.format(accuracy_sample))
		f.write('SK Validation Accuracy: {}\n'.format(sk_accuracy_sample))
		f.write('Validation MCC: {}\n'.format(mcc_sample))
		f.write('SK Validation MCC: {}\n'.format(sk_mcc_sample))
		f.write('Test Accuracy: {}\n'.format(accuracy))
		f.write('SK Test Accuracy: {}\n'.format(sk_accuracy))
		f.write('Test MCC: {}\n'.format(mcc))
		f.write('SK Test MCC: {}\n'.format(sk_mcc))

# close the file
file.close()
file_sk.close()
rf.close()
rf_sk.close()