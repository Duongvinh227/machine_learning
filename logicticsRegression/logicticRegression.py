import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1+np.exp(-z))

def accuracy_(p , labels_val):
    # p >= 0.5 = labels[1] / p < 0.5 = labels[0]
    result = np.zeros_like(p, dtype=int)
    result[p >= 0.5] = 1

    x = result == labels_val
    count_equal = np.count_nonzero(x)
    total = predict.size
    accuracy_ = count_equal/total
    return accuracy_

def predict(features, weights):
    z = np.dot(features, weights)
    result = sigmoid(z)
    return result

def loss_func(features, labels, weights):
    # cross_entropy
    """
    :praram features: 2926x15
    :praram labels: 2926x1 (0 , 1)
    :praram weights : 15x1
    :return loss:
    """
    n = len(labels)
    predictions = predict(features, weights)
    loss_class_1 = -labels*np.log(predictions)
    loss_class_0 = -(1 - labels)*np.log(1 - predictions)
    loss_total = loss_class_0 + loss_class_1

    return loss_total/n
def update_weights(features, labels, weights , learing_rate):
    """
    :praram features: 2925x14
    :praram labels: 2925x1 (0 , 1)
    :praram weights : 14x1
    :param learing_rate :
    :return new_weights:
    """
    n = len(labels)
    predictions = predict(features, weights)
    gd = np.dot(features.T,(predictions-labels))
    gd = gd/n
    gd = gd*learing_rate
    weights = weights - gd
    return weights

def train(features,labels,weights , learing_rate , epochs):
    loss_his = []

    for i in range(epochs):
        weights = update_weights(features, labels, weights, learing_rate)
        loss = loss_func(features, labels, weights)
        loss_his.append(loss)
        # print("epochs: ", i ,"loss:", loss )
    return weights, loss

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
def sklearn_logicticRegression(features_train, labels_train ,features_val, labels_val):
    #using scikit_learn

    # features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2,random_state=42)

    # Create and train the logistic regression model
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)

    # Reshape the labels to 1-dimensional arrays using ravel()
    labels_train = labels_train.ravel()
    labels_val = labels_val.ravel()

    # Create and train the logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(features_train_scaled, labels_train)

    # Make predictions on the validation set
    predictions_val = logistic_model.predict(features_val_scaled)

    # Calculate accuracy on the validation set
    accuracy = accuracy_score(labels_val, predictions_val)
    print("Validation accuracy:", accuracy)


if __name__=="__main__":

    df_logictics = pd.read_csv('data_for_logicticRegression.csv')
    df_logictics = df_logictics.dropna()
    df_logictics_train = df_logictics.sample(frac=0.8)
    df_logictics_val = df_logictics.drop(df_logictics_train.index)

    # values_true_2d = np.empty((0, 14), int)
    # values_flase_2d = np.empty((0, 14), int)
    #
    # for i, item in enumerate(df_logictics_train.values):
    #     if item[15] == 1.0:
    #         array_values = item[:14].reshape(1, -1)
    #         values_true_2d = np.vstack((values_true_2d, array_values))
    #     if item[15] == 0.0:
    #         array_values = item[:14].reshape(1, -1)
    #         values_flase_2d = np.vstack((values_flase_2d, array_values))

    features = df_logictics_train.iloc[:, :15].values
    labels = df_logictics_train.iloc[:, 15:].values
    features_val = df_logictics_val.iloc[:, :15].values
    labels_val = df_logictics_val.iloc[:, 15:].values

    weights_ = np.full((15, 1), 0.03)
    weights, bias = train(features, labels, weights_, 0.001, 100)
    predict = predict(features_val, weights)
    accuracy = accuracy_(predict, labels_val)
    print(accuracy)

    sklearn_logicticRegression(features, labels, features_val, labels_val)