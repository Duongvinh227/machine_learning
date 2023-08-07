import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))

    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if i == j else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def sklearn_decision_tree(dataset):
    X = dataset.iloc[:, [1 ,2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Decision Tree:", accuracy)
    display_confusion_matrix(y_test, y_pred)

if __name__=="__main__":
    dataset = pd.read_csv('decision_tree.csv')
    gender_map = {"Male": 0, "Female": 1}
    dataset["Gender"] = dataset["Gender"].map(gender_map)
    sklearn_decision_tree(dataset)
