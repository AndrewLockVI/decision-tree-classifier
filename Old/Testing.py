from Podtc import PseudoOptimalDecisionTreeClassifier
import numpy as np
##import plotly.express as px
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import load_digits
#import pandas as pd
#from keras.datasets import mnist
#from sklearn.metrics import accuracy_score
#from setuptools import setup

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

# train_X = train_X.reshape(-1, 784)
# test_X = test_X.reshape(-1, 784)

X_train = np.random.randint(1, 10, size=(3000, 784))

# Generate 60,000 labels for y_train (for example, labels between 1 and 10)
y_train = np.random.randint(1, 11, size=3000)

# train_X = [[2,5], [5,2], [3,4], [4,4], [5,5], [10, 10], [2,2], [12,12]]
# train_y = [1, 1 , 2, 1, 5, 2,1 ,3]

classifier = PseudoOptimalDecisionTreeClassifier(proportionToTrainOn=1, proportionToValidateSplits=1, proportionOfDimsToTrainOn=1, maxDepth=1);
classifier.fit(X_train, y_train)
exit()

y_pred = classifier.predict(test_X)
print("MY ACCURACY:")
print(accuracy_score(y_true=test_y, y_pred=y_pred))

classifier = DecisionTreeClassifier(max_depth=4)
classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)

print("SECOND ACCURACY:")
print(accuracy_score(y_true=test_y, y_pred=y_pred))


exit()

X = np.random.random((200, 2))
y = np.random.random((200)) * 10
y = y.round()
# y = (X[:,0] + X[:,1]) > .5


#clf = DecisionTreeClassifier()
#clf.fit(X,y)

classifier = PseudoOptimalDecisionTreeClassifier(proportionToTrainOn=1, proportionToValidateSplits=1, proportionOfDimsToTrainOn=1, maxDepth=2);

classifier.fit(X,y)


X_pred = np.random.random((20, 2)).round()

print(X_pred)
print(classifier.predict(X_pred))

# classifier.predict()
# print(classifier)



#scatter = px.scatter(x=X[:,0], y=X[:,1], color=y)
#scatter.show()
