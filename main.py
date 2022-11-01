import numpy as np
import pandas as pd
import Pfeature.pfeature as pf
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import lazypredict


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


train_directory = './train_set'
test_directory = './test_set'
pos_train = train_directory + '/train_po_cdhit.txt'
neg_train = train_directory + '/train_ne_cdhit.txt'
feature_train = feature_calc(pos_train, neg_train, aac)
X = feature_train.drop(['class'], axis=1)
Y = feature_train.iloc[:, -1].values.reshape(-1, 1)
#Y = feature_train['class'].copy()

X_train, X_test_sample, Y_train, Y_test_sample = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
X_test_sample = X_test_sample.values

mode = 'gini'
for depth in range(1, 20):
	classifier = DecisionTree(max_depth=depth, min_samples_split=2, min_impurity=1e-7, mode=mode)
	classifier.fit(X_train, Y_train)
	tree_file_name = 'test_results/tree{}-{}.txt'.format(classifier.max_depth, mode)
	classifier.print_tree_to_file(tree_file_name)

	pos_test = test_directory + '/test_po.txt'
	neg_test = test_directory + '/test_ne.txt'
	feature_test = feature_calc(pos_test, neg_test, aac)
	X_test = feature_test.iloc[:, :-1].values
	Y_test = feature_test.iloc[:, -1].values.reshape(-1, 1)

	Y_pred = classifier.predict(X_test)
	# print accuracy result to test_result
	accuracy = accuracy_score(Y_test, Y_pred)

	sk_classifier = DecisionTreeClassifier(criterion=mode, max_depth=classifier.max_depth, min_samples_split=classifier.min_samples_split)
	sk_classifier.fit(X_train, Y_train)
	sk_Y_pred = sk_classifier.predict(X_test)
	sk_accuracy = accuracy_score(Y_test, sk_Y_pred)


	accuracy_sample = accuracy_score(Y_test_sample, classifier.predict(X_test_sample))
	sk_accuracy_sample = accuracy_score(Y_test_sample, sk_classifier.predict(X_test_sample))

	accuracy_file_name = 'test_results/result{}-{}.txt'.format(classifier.max_depth, mode)
	with open(accuracy_file_name, 'w') as f:
		f.write('Sample Accuracy: {}\n'.format(accuracy_sample))
		f.write('SK Sample Accuracy: {}\n'.format(sk_accuracy_sample))
		f.write('Accuracy: {}\n'.format(accuracy))
		f.write('SK Accuracy: {}'.format(sk_accuracy))

