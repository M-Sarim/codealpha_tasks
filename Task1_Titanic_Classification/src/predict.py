import argparse
import pandas as pd
from joblib import load
from data_preprocessing import preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_proc = preprocess(df)
    model = load('models/titanic_model.pkl')
    df['Survived_Pred'] = model.predict(df_proc)
    df[['PassengerId', 'Survived_Pred']].to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
