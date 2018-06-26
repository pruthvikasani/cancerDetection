import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv("dataset.csv")


print "Dataset has been read successfully"

X = df.values[:,1:23]
Y = df.values[:,24]

print "Target variable and feature variables are selected"

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state =100)

print "Test data and Training data are split"

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)

#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               #max_depth=3, min_samples_leaf=5)
print "Gini index has been selected as the criteria"
clf_gini.fit(X_train, Y_train)

print "Model is fitted"

clf_gini.predict([[4,4,3,3,3,4,1,6,2,8,3,6,3,5,1,3,6,2,5,3,7,4]])
#single input prediction

print "Sample to be predicted is given"

y_pred = clf_gini.predict(X_test)
print(y_pred)

print "Accuracy is ", accuracy_score(Y_test,y_pred)*100


with open("clf_gini.dot", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f)
    
#dot -Tpdf clf_gini.dot -o clf_gini.pdf 

