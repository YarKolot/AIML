import pandas as pd
import numpy as np

def calc_r(xs, ys):
    rs = {}
    for col in xs.columns:
        x = xs[col].values
        y = ys.values.flatten()
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x)**2)) * np.sqrt(np.sum((y - mean_y)**2))
        r = numerator / denominator

        print(f"R for '{col}': {r:.4f}")
        rs[col] = r
    
    print()
    
    return rs

def plot(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

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
