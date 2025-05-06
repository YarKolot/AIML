# Dataset: Salary_dataset.csv
# Link to the dataset: https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression

import csv
import numpy as np
import matplotlib.pyplot as plt

def print_dataset_info(filename = "salary_dataset.csv"):
    indent = " " * 5

    fields = []
    rows = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        fields = next(csvreader)

        for row in csvreader:
            rows.append(row)
        
        print("Overal information about the dataset:")
        print(f"{indent}Filename: {filename}")
        print(f"{indent}Total no. of rows: {csvreader.line_num}")
        print(f"{indent}Total no. of fields: {len(fields)}")
        print(f"{indent}Rows with data: {csvreader.line_num - 1}")
        
        print()
    
        print("Structure of the dataset:")   
        print(f"{indent}Fields structure: {str(fields)}")
        print(f"{indent}Row structure: {str(rows[0])}")

    print('\nAll rows are:\n')

    for field in fields:
        print("%20s" % field, end=" ")
    print('\n')

    for row in rows:
        for col in row:
            print("%20s" % col, end=" ")
        print('\n')

# More readable and reliable version of the correlation coefficient calculation     
def calculate_correlation_coefficient_v1(rows, field_i1, field_i2):
    n = len(rows)
    sum_x = sum_y = 0.0

    for row in rows:
        x = float(row[field_i1])
        y = float(row[field_i2])
        sum_x += x
        sum_y += y

    avg_x = sum_x / n
    avg_y = sum_y / n

    sum_xy = sum_x2 = sum_y2 = 0.0
    for row in rows:
        x = float(row[field_i1])
        y = float(row[field_i2])
        dx = x - avg_x
        dy = y - avg_y
        sum_xy += dx * dy
        sum_x2 += dx ** 2
        sum_y2 += dy ** 2

    denominator = np.sqrt(sum_x2 * sum_y2)
    if denominator == 0:
        return 0.0

    return sum_xy / denominator

# More efficient version of the correlation coefficient calculation
def calculate_correlation_coefficient_v2(rows, field_i1, field_i2):
    n = len(rows)
    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0

    for row in rows:
        x = float(row[field_i1])
        y = float(row[field_i2])
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator

# Using numpy built-in function to calculate the correlation coefficient
def calculate_correlation_coefficient_builtin(rows, field_i1, field_i2):
    x = [float(row[field_i1]) for row in rows]
    y = [float(row[field_i2]) for row in rows]
    return np.corrcoef(x, y)[0, 1]

# n - 1 version of the standard deviation calculation
def standard_deviation(rows, field_i):
    n = len(rows)
    if n < 2:
        return 0.0

    sum_x = 0.0
    for row in rows:
        sum_x += float(row[field_i])

    avg_x = sum_x / n

    sum_x2 = 0.0
    for row in rows:
        dx = float(row[field_i]) - avg_x
        sum_x2 += dx ** 2

    return np.sqrt(sum_x2 / (n - 1))

def mean(rows, field_i):
    n = len(rows)
    if n == 0:
        return 0.0

    sum_x = 0.0
    for row in rows:
        sum_x += float(row[field_i])

    return sum_x / n

def predict(x, a, b):
    return a + b * x

def plot_data_and_regression_line(x_data, y_data, b, a):
    plt.scatter(x_data, y_data, color='blue', label='Data Points')
    plt.plot(x_data, [predict(x, a, b) for x in x_data], color='red', label='Regression Line')
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()
    plt.pause(0.001)

def main():
    filename = "salary_dataset.csv"
    indent = " " * 5
    
    fields = []
    rows = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        fields = next(csvreader)

        for row in csvreader:
            rows.append(row)
    
    print_dataset_info()
    
    print("\nCorrelation coefficient calculation:")
    correlation_coefficient_v1 = calculate_correlation_coefficient_v1(rows, 1, 2)
    print(f"{indent}Correlation coefficient between {fields[1]} and {fields[2]}: {correlation_coefficient_v1}")
    
    correlation_coefficient_v2 = calculate_correlation_coefficient_v2(rows, 1, 2)
    print(f"{indent}Correlation coefficient between {fields[1]} and {fields[2]}: {correlation_coefficient_v2}")
    
    correlation_coefficient_builtin = calculate_correlation_coefficient_builtin(rows, 1, 2)
    print(f"{indent}Correlation coefficient between {fields[1]} and {fields[2]}: {correlation_coefficient_builtin}")
    
    print("\nStandard deviation calculation:")
    x_standard_deviation = standard_deviation(rows, 1)
    y_standard_deviation = standard_deviation(rows, 2)
    print(f"{indent}Standard deviation of {fields[1]}: {x_standard_deviation}")
    print(f"{indent}Standard deviation of {fields[2]}: {y_standard_deviation}")
    print()
    
    # Calculating the slope of the regression line (b)
    b = correlation_coefficient_v1 * (y_standard_deviation / x_standard_deviation)
    print("Slope of the regression line (b): ", b)
    
    # Calculating the intercept of the regression line (a)
    a = mean(rows, 2) - b * mean(rows, 1)
    print("Intercept of the regression line (a): ", a)
    
    print()
    
    # Plotting the data and regression line
    x_data = [float(row[1]) for row in rows]
    y_data = [float(row[2]) for row in rows]
    plot_data_and_regression_line(x_data, y_data, b, a)
    
    # Predicting value of y for a given x
    x = float(input(f"Enter the value of {fields[1]} to predict {fields[2]}: "))
    y = predict(x, a, b)
    print(f"{indent}Predicted value of {fields[2]} for {fields[1]} = {x}: {np.round(y, 2)}")

if __name__ == "__main__":
    main()