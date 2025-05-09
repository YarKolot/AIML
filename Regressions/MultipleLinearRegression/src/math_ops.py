import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_r(xs, ys):
    rs = {}
    y = ys.values.flatten()
    mean_y = np.mean(y)
    
    for col in xs.columns:
        x = xs[col].values
        mean_x = np.mean(x)

        numerator = np.dot(x - mean_x, y - mean_y)
        denominator = np.sqrt(np.dot(x - mean_x, x - mean_x)) * np.sqrt(np.dot(y - mean_y, y - mean_y))

        r = numerator / denominator

        print(f"R for '{col}': {r:.4f}")
        rs[col] = r
        
    print()
    
    return rs

def plot(df):
    x_columns = df.columns[:-1]
    y_col = df.columns[-1]
    drawn_columns = []

    while True:
        print("Available X columns:")
        for idx, col in enumerate(x_columns):
            print(f"{idx}: {col}")
        
        print()
        
        while True:
            print(f"Columns already plotted: {', '.join([x_columns[i] for i in drawn_columns])}")
            try:
                x_i = int(input("Enter the index of the X column: "))
                if 0 <= x_i < len(x_columns) and x_i not in drawn_columns:
                    break
                elif x_i in drawn_columns:
                    print(f"Graph for '{x_columns[x_i]}' has already been drawn. Choose another column.")
                else:
                    print("Index out of range. Try again.")
            except ValueError:
                print("Invalid input. Enter an integer.")
        
        drawn_columns.append(x_i)
        x_col = x_columns[x_i]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[x_col], y=df[y_col])
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        print()

        continue_choice = input("Do you want to plot another graph? (y/n): ").strip().lower()
        
        print()
        if continue_choice == 'n':
            print("No more graphs will be plotted.")
            break

def calc_b1(xs, ys):
    b1 = {}
    y = ys.values.flatten()
    mean_y = np.mean(y)
    
    for col in xs.columns:
        x = xs[col].values
        
        mean_x = np.mean(x)

        numerator = np.dot(x - mean_x, y - mean_y)
        denominator = np.dot(x - mean_x, x - mean_x)
        b1[col] = numerator / denominator

    return b1

def calc_b0(xs, ys, b1):
    y = ys.values.flatten()
    mean_y = np.mean(y)
    res = sum(b1[col] * np.mean(xs[col].values) for col in xs.columns)
    b0 = mean_y - res
    return b0

def normal_calc_betas(xs, ys):
    X = xs.values
    y = ys.values.flatten()

    X = np.column_stack((np.ones(X.shape[0]), X))

    Xt = X.T
    XtX = Xt @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = Xt @ y
    b = XtX_inv @ Xty
    
    feature_names = ['intercept'] + list(xs.columns)
    
    return pd.DataFrame({'beta': b}, index=feature_names)

def calc_y(xs, b0, b1):
    y = b0 + sum(b1[col] * xs[col].values for col in xs.columns)
    return y

def calc_predict(test_xs, b0, b1):
    b1_vector = np.array([b1[col] for col in test_xs.columns])
    X = test_xs.values
    preds = b0 + X @ b1_vector
    return pd.Series(preds, index=test_xs.index)

def compare_predictions(test_ys, predictions):
    comparison = pd.DataFrame({
        'Actual': test_ys.values.flatten(),
        'Predicted': predictions
    })
    
    comparison['Difference'] = comparison['Actual'] - comparison['Predicted']
    return comparison

def calc_r_squared(test_ys, predictions):
    ss_total = np.sum((test_ys.values.flatten() - np.mean(test_ys.values.flatten())) ** 2)
    ss_residual = np.sum((test_ys.values.flatten() - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def calc_rmse(test_ys, predictions):
    rmse = np.sqrt(np.mean((test_ys.values.flatten() - predictions) ** 2))
    return rmse