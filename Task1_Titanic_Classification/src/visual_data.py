import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    os.makedirs('visuals', exist_ok=True)
    df = pd.read_csv('data/train.csv')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)

    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title('Survival by Gender')
    plt.savefig('visuals/survival_by_gender.png')
    plt.clf()

    sns.countplot(x='Survived', hue='Pclass', data=df)
    plt.title('Survival by Class')
    plt.savefig('visuals/survival_by_class.png')
    plt.clf()

    sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
    plt.title('Age Distribution by Survival')
    plt.savefig('visuals/age_distribution.png')
    plt.clf()

    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    corr = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.savefig('visuals/feature_correlation.png')
    plt.clf()

    print("Visualizations saved in 'visuals/'")

if __name__ == '__main__':
    main()
