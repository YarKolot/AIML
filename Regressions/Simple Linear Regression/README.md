# Project: Salary Prediction using Linear Regression

This project applies a simple linear regression model to predict salary based on years of professional experience. Using a dataset containing both variables, the model computes the correlation coefficient, standard deviations, and predicts salary values. Additionally, the program visualizes the regression line and data distribution.

## 📁 Project Structure

The structure is simple:

1. **salary_dataset.csv** – CSV file containing the dataset (YearsExperience, Salary)
2. **main.py** – Main Python file containing all the code and logic

## ⚙️ Installation & Running

To run the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ykz737/AIML/Regressions/Simple Linear Regression.git
cd Simple Linear Regression
```

2. Install dependencies

```bash
pip install numpy matplotlib
```

3. Run the script

```bash
python main.py
```

## 🚀 How It Works

### 1. **Data Loading**

The project begins by loading a CSV file, `salary_dataset.csv`, containing two columns:

- **YearsExperience** – The number of years of experience.
- **Salary** – The corresponding salary for each experience value.

The dataset is parsed and displayed in a readable format for further analysis.

### 2. **Correlation Calculation**

To understand the relationship between **YearsExperience** and **Salary**, the correlation coefficient is calculated using three methods:

- **Method 1**: A manual approach that sums the values and computes the correlation coefficient.
- **Method 2**: A more efficient method, minimizing redundant calculations.
- **Method 3**: The built-in NumPy function (`np.corrcoef`), which offers a quick and reliable computation.

These methods help in determining the strength and direction of the relationship between years of experience and salary.

### 3. **Standard Deviation**

The program computes the **standard deviation** for both **YearsExperience** and **Salary**. Standard deviation is essential to understand how spread out the data points are. It is calculated using the sample standard deviation formula, providing insights into the variability of both variables.

### 4. **Regression Model**

With the calculated correlation coefficient and standard deviations, the model computes the **slope (b)** and **intercept (a)** for the regression line:

- **Slope (b)** is calculated as:
  \[
  b = \text{correlation coefficient} \times \left( \frac{\text{Standard Deviation of Salary}}{\text{Standard Deviation of Experience}} \right)
  \]

- **Intercept (a)** is calculated as:
  \[
  a = \text{Mean of Salary} - b \times \text{Mean of Experience}
  \]

This regression line provides the best fit, helping predict salary based on years of experience.

### 5. **Visualization**

The script generates a plot that shows:

- The **data points** (Years of Experience vs. Salary).
- The **regression line**, which represents the predicted relationship between experience and salary.

This visualization helps in understanding the linear relationship and how well the model fits the data, providing a clear visual representation of the prediction.

### 6. **Prediction**

Once the regression model is established, you can input a value for **YearsExperience**, and the model will predict the corresponding **Salary** using the regression equation. This makes it easy to estimate the salary for a specific experience level.