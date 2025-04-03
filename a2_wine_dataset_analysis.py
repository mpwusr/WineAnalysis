# this is an example exploration of the UCI Wine Quality Dataset, focusing on data summary, regression, and classification
# https://archive.ics.uci.edu/ml/datasets/wine+quality
# here, we will look at winequality-red.csv
#

import matplotlib.pyplot as plt
# import the needed packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# If you use the conda package manager, the graphviz binaries and the python package can be installed with
# conda install python-graphviz
# Alternatively binaries for graphviz can be downloaded from the graphviz project homepage, 
# and the Python wrapper installed from pypi with pip install graphviz.

# temp from sklearn.tree import export_graphviz
# temp from pydotplus import graph_from_dot_data

#
# load the data
#

# note: the default read_csv() assumes the separator is ',', so if your data is something different, you must specify it
df = pd.read_csv("a2_winequality-red.csv", sep=";", encoding='utf-8')


#
# DATA EXPLORATION
#


# summary of the dataset
print("\nsummary of the dataset")
df.info()

# view first five examples
print("\nview first five examples")
print(df.head())

# view the number of rows and columns
print("\nview the number of rows and columns:")
print(df.shape)

# view the quality variable value counts
print("\nview the quality variable value counts:")
print(df['quality'].value_counts())

# view a barplot of the quality (categorical variables)
print("\nview a barplot of the quality (categorical variables)")
sns.countplot(x='quality', data=df, order=df['quality'].value_counts().index)
plt.show()

# example histogram for quantitative (not categorical) variables
print("\nexample histogram for quantitative (not categorical) variables")
plt.hist('fixed acidity',data=df,bins=10)

plt.xlabel('Fixed Acidity')
plt.ylabel('Count')
plt.title('Histogram of Fixed Acidity')
plt.xlim(4,16)
plt.ylim(0, 600)
plt.grid(True)
plt.show()

# example variable statistics: mean, median, mode
print("\nexample variable statistics: mean, median, mode")
print('Mean',round(df['quality'].mean(),2))
print('Median',df['quality'].median())
print('Mode',df['quality'].mode()[0])

#
# SIMPLE LINEAR REGRESSION
#
print("\nbegin SIMPLE LINEAR REGRESSION")
train_data,test_data = train_test_split(df,train_size=0.8,random_state=1234)
reg = linear_model.LinearRegression()
x_train = np.array(train_data['fixed acidity']).reshape(-1,1)
y_train = np.array(train_data['quality']).reshape(-1,1)
reg.fit(x_train,y_train)

x_test = np.array(test_data['fixed acidity']).reshape(-1,1)
y_test = np.array(test_data['quality']).reshape(-1,1)
pred = reg.predict(x_test)
print('linear model')
mean_squared_error = metrics.mean_squared_error(y_test,pred)
print('Squared mean error', round(np.sqrt(mean_squared_error),2))
print('R squared training',round(reg.score(x_train,y_train),3))
print('R squared testing',round(reg.score(x_test,y_test),3) )
print('intercept',reg.intercept_)
print('coefficient',reg.coef_)

plt.scatter(x_test, y_test, color= 'darkgreen', label = 'data')
plt.plot(x_test, reg.predict(x_test), color='red', label= ' Predicted Regression line')
plt.xlabel('fixed acidity')
plt.ylabel('quality')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

#
# MULTIPLE REGRESSION
#

print("\nbegin Multiple Regression")
features1 = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
reg = linear_model.LinearRegression()
reg.fit(train_data[features1],train_data['quality'])
pred = reg.predict(test_data[features1])
print('multiple regression')
mean_squared_error = metrics.mean_squared_error(y_test,pred)
print('mean squared error(MSE)', round(np.sqrt(mean_squared_error),2))
print('R squared training',round(reg.score(train_data[features1],train_data['quality']),3))
print('R squared testing', round(reg.score(test_data[features1],test_data['quality']),3))
print('Intercept: ', reg.intercept_)
print('Coefficient:', reg.coef_)


#
# CLASSIFICATION
#
# NOTE: useful --> https://scikit-learn.org/stable/user_guide.html
#

print("\nbegin CLASSIFICATION")
# divide data into training/test sets
y = df.quality
X = df.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)


#
# decision tree
#

print("\nbegin decision tree")
# train/test decision tree model
tree_model = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)

train_score = tree_model.score(X_train, y_train)
test_score = tree_model.score(X_test, y_test)

print("decision tree training accuracy: ", train_score)
print("decision tree test accuracy: ", test_score)

print("let's give the model an example to see what class label it predicts:")
# remember, here are the variable names ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
print(tree_model.predict([[7.4,	0.7,	0,	1.9,	0.076,	11,	34,	0.9978,	3.51,	0.56,	9.4]]))
print("and here are the probabilities for each of the class labels:")
print(tree_model.predict_proba([[7.4, 0.7,	0,	1.9,	0.076,	11,	34,	0.9978,	3.51,	0.56,	9.4]]))

# temp print("create a decision tree and store it as a file")
# temp dot_data = export_graphviz(tree_model, filled=True, rounded=True, feature_names=['fixed acidity', 'volatile acidity', 'citric acid',	'residual sugar', 'chlorides',	'free sulfur dioxide',	'total sulfur dioxide',	'density',	'pH',	'sulphates',	'alcohol'], class_names=['3', '4', '5', '6', '7', '8'])
# temp graph = graph_from_dot_data(dot_data)
# temp graph.write_png('wine_tree.png')


#
# softmax regression
#

print("\nbegin softmax regression")
softmax_reg = linear_model.LogisticRegression(solver='lbfgs', C=10, random_state=42, multi_class='multinomial')
softmax_reg.fit(X_train,y_train)

train_score = softmax_reg.score(X_train, y_train)
test_score = softmax_reg.score(X_test, y_test)

print("logistic regression training accuracy: ", train_score)
print("logistic regression test accuracy: ", test_score)

print("let's give the model an example to see what class label it predicts:")
print(softmax_reg.predict([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]))
print("and here are the probabilities for each of the class labels:")
print(softmax_reg.predict_proba([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]))

print("\nbegin confusion matrix")


# confusion matrix using graphviz
# temp print("\nbegin confusion matrix graphviz")
# temp predictions = softmax_reg.predict(X_test)
# temp cm = metrics.confusion_matrix(y_test, predictions)
# temp plt.figure(figsize=(9,9))
# temp sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
# temp plt.ylabel('Actual label');
# temp plt.xlabel('Predicted label');
# temp all_sample_title = 'Accuracy Score: {0}'.format(test_score)
# temp plt.title(all_sample_title, size = 15);
# temp plt.show()

# confusion matrix using confusionmatrixdisplay
predictions = softmax_reg.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions)
print("you can also just print the cm to the command line:")
print(cm)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# confusion matrix plain
# plot_confusion_matrix is deprecated
# disp = metrics.plot_confusion_matrix(softmax_reg, X_test, y_test, display_labels=None, cmap=plt.cm.Blues)
# disp.ax_.set_title('Accuracy Score: {0}'.format(test_score))
# print('Accuracy Score: {0}'.format(test_score))
# print(disp.confusion_matrix)
# plt.show()


print("done")
