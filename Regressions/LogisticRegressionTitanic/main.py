import pandas as pd
import numpy as np
import src.test_data_ops as tdo
import src.test_model_ops as tmodo

LEARNING_RATE = 0.05
EPOCHS = 5000

def main():
    train_file = "data/train.csv"
    test_file = "data/tested.csv"
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    combined = [train, test]
    
    for i in range(len(combined)):
        # Checking out data
        tdo.print_info(combined[i], i, False)
        
        # Preparing dataset
        combined[i] = tdo.prepare_data(combined[i])
        
        # Checking out prepared data
        tdo.print_info(combined[i], i, True)
        
    # Splitting into X and y
    X_train, y_train = tdo.devide_to_Xy(combined[0])
    
    # Generating random weights and bias
    w, b = tdo.gen_weights(X_train)
    w = w.reshape(-1, 1)

    # Training the model
    print("Training started")
    w, b = tmodo.train_model(X_train, y_train, w, b, LEARNING_RATE, EPOCHS)
    print("Training finished")
    
    # Making predictions
    X_test, y_test = tdo.devide_to_Xy(combined[1])
    y_pred = tmodo.predict(X_test, w, b)
    y_pred = (y_pred >= 0.5).astype(int)
    
    # Evaluating the model
    accuracy, precision, recall, f1 = tmodo.evaluate_model(y_test, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
