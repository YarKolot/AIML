import csv
import os
import pandas as pd

def read(filename):
    fields = []
    rows = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        fields = [field.strip().lower().replace(' ', '_') for field in fields]

        for row in csvreader:
            for i in range(len(row)):
                row[i] = row[i].strip().lower().replace(' ', '_')
                if row[i] == 'true':
                    row[i] = 1
                elif row[i] == 'false':
                    row[i] = 0
                elif row[i] == 'yes':
                    row[i] = 1
                elif row[i] == 'no':
                    row[i] = 0
                    
                try:
                    row[i] = float(row[i])
                except ValueError:
                    pass
                
            rows.append(row)
            
    return fields, rows

def print_info(filename, fields, rows):
    indent = " " * 5

    print("Overal information about the dataset:")
    print(f"{indent}Filename: {os.path.basename(filename)}")
    print(f"{indent}Total no. of rows: {len(rows) + 1}")
    print(f"{indent}Total no. of fields: {len(fields)}")
    print(f"{indent}Rows with data: {len(rows)}")
    
    print()
    
    print("Structure of the dataset (first 3 rows):")   
    print(f"{indent}Fields: ", end="")
    for field in fields:
        print(f"'{field}'; ", end="")
    print()
    print(f"{indent}Row structure: {str(rows[0])}")
    print()
        
def make_df(fields, rows):
    df = pd.DataFrame(rows, columns=fields)
    df = df.dropna(how='any')

    object_cols = df.select_dtypes(include=['object'])

    if not object_cols.empty:
        dummies = pd.get_dummies(object_cols, drop_first=True)
        df = pd.concat([dummies, df.select_dtypes(exclude=['object'])], axis=1)
    else:
        df = df.select_dtypes(exclude=['object'])

    df = df.dropna(how='any')
    return df

def make_x_y(df):
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns (features and target)")

    xs = df.iloc[:, :-1]
    ys = df.iloc[:, -1]
    
    return xs, ys
