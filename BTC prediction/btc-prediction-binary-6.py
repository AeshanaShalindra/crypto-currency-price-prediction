import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import tree
from sklearn import metrics
import joblib

#https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/

dataset = pd.read_csv("Outputs/combined_final_data_2-btc.csv")
dataset.drop(dataset.columns[[0, 1, 2, 3]], axis=1, inplace=True)
dataset['Change %'] = dataset['Change %'].astype(int)
# Printing first 5 records of the dataset
#print(dataset.head(5))
#print(dataset.shape)
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

X = dataset.drop(['Change %'], axis=1)
Y = dataset['Change %']
#print(X)
#print(Y)
# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.5, test_size=0.5, random_state=0)

#SVM – Support vector Machine
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
joblib.dump(model_SVR, 'regression_models/svm_model_binary.joblib')
Y_pred = model_SVR.predict(X_valid)
Y_pred = (np.rint(Y_pred)).astype(int)
print(mean_absolute_percentage_error(Y_valid, Y_pred))
mae = metrics.mean_absolute_error(Y_valid, Y_pred)
mse = metrics.mean_squared_error(Y_valid, Y_pred)
r2 = metrics.r2_score(Y_valid, Y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

fig, ax = plt.subplots()
ax.scatter(Y_pred, Y_valid, edgecolors=(0, 0, 1))
ax.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'r--', lw=3)
ax.set_xlabel('Predicted-SVM')
ax.set_ylabel('Actual')
plt.show()

#Random Forest Regression
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
joblib.dump(model_RFR, 'regression_models/rf_model_binary.joblib')
Y_pred = model_RFR.predict(X_valid)
Y_pred = (np.rint(Y_pred)).astype(int)
print(Y_pred)
print(mean_absolute_percentage_error(Y_valid, Y_pred))

mae = metrics.mean_absolute_error(Y_valid, Y_pred)
mse = metrics.mean_squared_error(Y_valid, Y_pred)
r2 = metrics.r2_score(Y_valid, Y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

fig, ax = plt.subplots()
ax.scatter(Y_pred, Y_valid, edgecolors=(0, 0, 1))
ax.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'r--', lw=3)
ax.set_xlabel('Predicted-RF')
ax.set_ylabel('Actual')
plt.show()

#Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
joblib.dump(model_LR, 'regression_models/lr_model_binary.joblib')
Y_pred = model_LR.predict(X_valid)
Y_pred = (np.rint(Y_pred)).astype(int)
print(Y_pred)
print(mean_absolute_percentage_error(Y_valid, Y_pred))

mae = metrics.mean_absolute_error(Y_valid, Y_pred)
mse = metrics.mean_squared_error(Y_valid, Y_pred)
r2 = metrics.r2_score(Y_valid, Y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

fig, ax = plt.subplots()
ax.scatter(Y_pred, Y_valid, edgecolors=(0, 0, 1))
ax.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'r--', lw=3)
ax.set_xlabel('Predicted-LR')
ax.set_ylabel('Actual')
plt.show()



#https://mljar.com/blog/extract-rules-decision-tree/

def get_rules(tree, feature_names, class_names):
	tree_ = tree.tree_
	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]

	paths = []
	path = []

	def recurse(node, path, paths):

		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			p1, p2 = list(path), list(path)
			p1 += [f"({name} <= {np.round(threshold, 3)})"]
			recurse(tree_.children_left[node], p1, paths)
			p2 += [f"({name} > {np.round(threshold, 3)})"]
			recurse(tree_.children_right[node], p2, paths)
		else:
			path += [(tree_.value[node], tree_.n_node_samples[node])]
			paths += [path]

	recurse(0, path, paths)

	# sort by samples count
	samples_count = [p[-1][1] for p in paths]
	ii = list(np.argsort(samples_count))
	paths = [paths[i] for i in reversed(ii)]

	rules = []
	for path in paths:
		rule = "if "

		for p in path[:-1]:
			if rule != "if ":
				rule += " and "
			rule += str(p)
		rule += " then "
		if class_names is None:
			rule += "response: " + str(np.round(path[-1][0][0][0], 3))
		else:
			classes = path[-1][0][0]
			l = np.argmax(classes)
			rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
		rule += f" | based on {path[-1][1]:,} samples"
		rules += [rule]

	return rules

regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
model = regr.fit(X_train, Y_train)

# Print rules
print("set of human readable rule ...............")
rules = get_rules(regr, X_train.columns, None)
for r in rules:
    print(r)