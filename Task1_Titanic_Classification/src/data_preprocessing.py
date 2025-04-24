import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Sex_male', 'Embarked_Q', 'Embarked_S']
    return df[features + (['Survived'] if 'Survived' in df.columns else [])]

def main():
    os.makedirs('data/processed', exist_ok=True)
    df = pd.read_csv('data/train.csv')
    df_proc = preprocess(df)
    train, val = train_test_split(df_proc, test_size=0.2, random_state=42, stratify=df_proc['Survived'])
    train.to_csv('data/processed/train_processed.csv', index=False)
    val.to_csv('data/processed/val_processed.csv', index=False)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
