import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
X = X.astype(np.float32)  # Ensure data is in float64 format for the classifier
y = y.astype(np.int32)    # Ensure target labels are integers

accLs = []
xVals = []

SEED = 110

np.random.seed(SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
y_train = np.array(y_train)
X_train = np.array(X_train)
X_test = np.array(X_test)

for i in range(1, 11001):
    if (i <= 100 and i % 10 == 0) or ((i) % 500 == 0) or (i == 1) or (i <= 500 and i % 50 == 0):
        X_train_pca = np.array(X_train).copy()
        X_test_pca = np.array(X_test).copy()



        num_samples = i
        random_indices = np.random.choice(len(X_train_pca) - 1, num_samples, replace=False)


        X_train_pca = X_train_pca[random_indices]
        y_train_current = y_train[random_indices]

        clf = DecisionTreeClassifier(100)

        start_time = time.time()
        clf.fit(X_train_pca, len(X_train_pca), y_train_current, len(X_train_pca[0]))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{i}: Time taken: {elapsed_time:.4f} seconds")
        y_pred = clf.predict(X_test_pca, len(X_test_pca), len(X_test_pca[0]))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of Decision Tree Classifier on MNIST: {accuracy * 100:.2f}%")

        accLs.append(accuracy)
        xVals.append(i)

df = pd.DataFrame(accLs, xVals)
df.to_csv('output.csv') 


plt.plot(xVals, accLs)
plt.show()
