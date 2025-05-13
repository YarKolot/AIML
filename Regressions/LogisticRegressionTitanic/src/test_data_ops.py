import pandas as pd
import numpy as np

def print_info(df, i, is_prepared):
    print(f"DataFrame {i} info{' (tested)' if is_prepared else ''}:")
    print(df.info())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print()
    
def prepare_data(df):
    # Making Title column
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    }
    df['Title'] = df['Title'].apply(lambda x: x if x in title_map else 'Rare')
    df['Title'] = pd.Categorical(df['Title'], categories=['Rare', 'Mr', 'Miss', 'Mrs', 'Master'], ordered=True)

    # Filling missing Age values
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
        
    # Deleting rows with missing Embarked and Fare values
    df.dropna(subset=['Fare', 'Embarked'], inplace=True)
        
    # Deleting useless columns
    df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True)
        
    # Creating dummy variables
    df = pd.get_dummies(df, columns=['Title', 'Sex', 'Embarked'], drop_first=True)
        
    # Creating FamilySize column out of SibSp and Parch
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(columns=['SibSp', 'Parch'], inplace=True)
    
    # Normalizing Age, Fare, and FamilySize columns
    df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()
    df['FamilySize'] = (df['FamilySize'] - df['FamilySize'].mean()) / df['FamilySize'].std()
        
    # Putting Survived column at the end
    survived = df.pop('Survived')
    df['Survived'] = survived
    
    # Converting columns to float64 so there won't be any problems with training
    df = df.astype('float64')
    
    return df

def devide_to_Xy(df):
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy().reshape(-1, 1)
    return X, y

def gen_weights(X):
    np.random.seed(1)
    n = X.shape[1]
    rand_w = np.random.randn(n, 1)
    rand_b = np.random.randn(1)[0]
    return rand_w, rand_b
