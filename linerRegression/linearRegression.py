import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def predict(x , weights , bias):
    return x*weights + bias

def loss_function(x, y, weights, bias):
    # MSE Mean Squared Error
    n = len(x)
    sum_error = 0;
    for i in range(n):
        sum_error += (y[i] - (weights*x[i]+bias))**2
    return sum_error/n

def update_weight(x , y, weights, bias , learing_rate):
    # gradient_Descent
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*x[i]*(y[i] - (x[i]*weights+bias))
        bias_temp += -2*(y[i] - (x[i]*weights+bias))

    weights -= (weight_temp/n)*learing_rate
    bias -= (bias_temp/n)*learing_rate

    return weights, bias

def train(x,y,weights, bias , learing_rate , epochs):
    loss_history = []

    for i in range(epochs):
        weights, bias = update_weight(x,y,weights, bias, learing_rate)
        loss = loss_function(x,y,weights, bias)
        loss_history.append(loss)
        print("epochs:", i,"loss:",loss)
    return weights, bias

def sklearn_linearRegression(x,y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x,y)

    weights = model.coef_[0]
    bias = model.intercept_

    return weights, bias

if __name__=="__main__":
    # Load data
    df_linearRegression = pd.read_csv("data_for_linearRegression.csv")
    df_x = df_linearRegression["x"].values
    df_y = df_linearRegression["y"].values

    # plt.scatter(df_x, df_y)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Dữ liệu cho Linear Regression")
    # plt.show()

    # weights, bias = train(df_x, df_y, 0.03, 0.0014, 0.0001, 30)
    weights, bias = sklearn_linearRegression(df_x, df_y)

    # Prediction on test data
    y_pred = predict(df_x, weights, bias)

    # Plot the regression line with actual data pointa
    plt.plot(df_x, df_y, 'o', label='Actual values')
    plt.plot(df_x, y_pred, label='Predicted values')
    plt.xlabel('Test input')
    plt.ylabel('Test Output or Predicted output')
    plt.legend()
    plt.show()