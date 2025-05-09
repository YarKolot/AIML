
# Multiple Linear Regression (MLR) Implementation

This is my implementation of Multiple Linear Regression (MLR). I created this project for training and learning the fundamentals of MLR. It has been a great learning experience, and I gained deeper insights into how linear regression works and how to implement it from scratch.

## Accuracy
The model achieved an R² score of **90%** on the dataset, which shows that it explains a significant portion of the variance in the data.

## How to Run

To test the implementation locally, follow these steps:

1. **Clone this repository:**
   ```bash
   git clone <repository_url>
   ```

2. **Install dependencies:**
   Make sure you have Python 3 installed, and then install the necessary packages using `pip`. You can install the dependencies with:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, here are the dependencies you will need to manually install:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`

3. **Run the project:**
   Once the dependencies are installed, you can run the `main.py` file:
   ```bash
   python main.py
   ```

   The main file will execute the script, perform the regression, and show you the results including the accuracy metrics, predictions, and comparisons.

## Features

- **Multiple Linear Regression:** Implemented the algorithm from scratch.
- **Model evaluation:** Includes R² score calculation and Mean Squared Error (MSE).
- **Data visualization:** Ability to visualize relationships between features and target variable.
- **Prediction comparison:** Compare actual values against predicted values for better analysis.
