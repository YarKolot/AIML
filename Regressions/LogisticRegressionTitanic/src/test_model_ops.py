import pandas as pd
import numpy as np

# Fucntion to calculate the linear regression value (z)
def calc_z(X, w, b):
    return np.dot(X, w) + b

# Function to calculate the sigmoid value
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to calculate gradients
def calc_gradients(X, y, y_hat):
    n = len(y)
    error = y_hat - y
    grad_w = (1 / n) * np.dot(X.T, error)
    grad_b = (1 / n) * np.sum(error)
    return grad_w, grad_b

# Function to update weights and bias
def update_weights(w, b, grad_w, grad_b, learning_rate=0.01):
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    return w, b

# Function to calculate the loss
def calc_loss(y, y_hat):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss

# Main function to train the model
def train_model(X, y, w, b, learning_rate, epochs):
    import matplotlib.pyplot as plt
    loss_history = []
    for epoch in range(epochs):
        z = calc_z(X, w, b)
        y_hat = sigmoid(z)
        loss = calc_loss(y, y_hat)
        loss_history.append(loss)
        grad_w, grad_b = calc_gradients(X, y, y_hat)
        w, b = update_weights(w, b, grad_w, grad_b, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    print("Final Weights:")
    print(w)
    print("Final Bias:", b)

    plt.plot(range(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()
    
    return w, b

def predict(X, w, b):
    z = calc_z(X, w, b)
    y_hat = sigmoid(z)
    return y_hat

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = (y_pred >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1