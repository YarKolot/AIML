import src.data_ops as do
import src.math_ops as mo

def main():
    filename = "data/train_energy_data.csv"
    fields, rows = do.read(filename)
    
    do.print_info(filename, fields, rows)
    
    df = do.make_df(fields, rows)
    
    xs, ys = do.make_x_y(df)
    
    mo.calc_r(xs, ys)
    
    mo.plot(df)

if __name__ == "__main__":
    main()