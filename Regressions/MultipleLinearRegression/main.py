import src.data_ops as do
import src.math_ops as mo

def main():
    training_file = "data/train_energy_data.csv"
    test_file = "data/test_energy_data.csv"
    fields, rows = do.read(training_file)
    
    # do.print_info(filename, fields, rows)
    
    df = do.make_df(fields, rows)
    
    xs, ys = do.make_x_y(df)
    
    mo.calc_r(xs, ys)
    
    b1 = mo.calc_b1(xs, ys)
    
    b0 = mo.calc_b0(xs, ys, b1)
    
    test_fields, test_rows = do.read(test_file)
    
    test_df = do.make_df(test_fields, test_rows)
    
    test_xs, test_ys = do.make_x_y(test_df)
    
    predictions = mo.calc_predict(test_xs, b0, b1)
    
    print("Compare:")
    print(mo.compare_predictions(test_ys, predictions))
    
    print()
    
    print("R-squared:")
    print(mo.calc_r_squared(test_ys, predictions))
    
    print()
    
    print("RMSE:")
    print(mo.calc_rmse(test_ys, predictions))
    

if __name__ == "__main__":
    main()