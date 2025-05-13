
# Titanic Survival Prediction with Logistic Regression

This is my implementation of Logistic Regression from scratch. It was trained on popular Titanic Dataset and predicts whether a passenger survived or not based on various features. This project was created for training and learning the fundamentals of Logistic Regression.

### Key Features of the Model:
- **Data Cleaning and Preparation**: The dataset contains missing values and irrelevant columns. These are handled by:
    - Filling missing age values based on the median of the respective class and sex group.
    - Removing rows with missing 'Fare' or 'Embarked' values.
    - Dropping unnecessary columns like 'PassengerId', 'Ticket', 'Cabin', and 'Name'.
- **Feature Engineering**: Additional features are derived, such as the 'FamilySize' (based on SibSp and Parch), and normalization is applied to key features such as Age, Fare, and FamilySize to improve model training.
- **Training**: The model was trained using logistic regression with a learning rate of 0.05 and for 5000 epochs to optimize the weights. The optimization was done using gradient descent.
- **Evaluation**: The model was evaluated based on accuracy, precision, recall, and F1 score.

## Results:
The model achieved **accuracy of 93.5%** on the Titanic dataset.

### Model Performance:
- **Accuracy**: 93.53%
- **Precision**: 88.82%
- **Recall**: 94.08%
- **F1 Score**: 91.37%

## Files:
- **train.csv**: The training dataset containing passenger information.
- **tested.csv**: The test dataset used to evaluate the model.
- **main.py**: The main script for training and evaluating the model.
- **src/test_data_ops.py**: Contains functions for data preprocessing, such as cleaning and feature engineering.
- **src/test_model_ops.py**: Contains functions for logistic regression, loss calculation, and gradient descent.

## How to Run

To test the implementation locally, follow these steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/YarKolot/AIML/tree/main/Regressions/LogisticRegressionTitanic
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
   - `scikit-learn`

3. **Run the project:**
   Once the dependencies are installed, you can run the `main.py` file:
   ```bash
   python main.py
   ```

   The main file will execute the script, perform the regression, and show you the results including the accuracy metrics, predictions, and comparisons.
