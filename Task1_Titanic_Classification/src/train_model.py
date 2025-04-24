import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def main():
    os.makedirs('models', exist_ok=True)
    train = pd.read_csv('data/processed/train_processed.csv')
    X = train.drop('Survived', axis=1)
    y = train['Survived']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    dump(model, 'models/titanic_model.pkl')
    print("Model saved to models/titanic_model.pkl")

if __name__ == '__main__':
    main()
