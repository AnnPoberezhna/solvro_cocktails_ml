import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df: pd.DataFrame):
    """Perform Exploratory Data Analysis on the cocktail dataset."""

    # Distribution of cocktail categories
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
    plt.title('Distribution of Cocktail Categories')
    plt.xticks(rotation=45)
    plt.ylabel("Number of Cocktails")
    plt.tight_layout()
    plt.savefig('./result/cocktail_category_distribution.png')

    # Alcoholic vs Non-Alcoholic
    plt.figure(figsize=(6, 4))
    alcoholic_counts = df['alcoholic'].value_counts()

    # Rename the index values if both types are present
    if len(alcoholic_counts) > 1:
        alcoholic_counts.index = ['Alcoholic', 'Non-Alcoholic']
    elif alcoholic_counts.index[0] == 1:
        alcoholic_counts.index = ['Alcoholic']
    else:
        alcoholic_counts.index = ['Non-Alcoholic']

    alcoholic_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'][:len(alcoholic_counts)])
    plt.title("Distribution of Alcoholic vs Non-Alcoholic Cocktails")
    plt.xlabel("Type")
    plt.ylabel("Number of Cocktails")
    plt.savefig('./result/alcohol_distribution.png')
    plt.xticks(rotation=0)

    plt.show()