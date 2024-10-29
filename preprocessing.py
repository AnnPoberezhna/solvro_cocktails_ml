import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def preprocess_ingredients(ingredients_df):
    """
        Preprocess ingredient names using TF-IDF vectorization.
    """
    tfidf = TfidfVectorizer()
    ingredient_names_tfidf = tfidf.fit_transform(ingredients_df['ingredient_name']).toarray()
    ingredient_names_df = pd.DataFrame(ingredient_names_tfidf, columns=tfidf.get_feature_names_out())
    return ingredient_names_df

def preprocess_measures(ingredients_df):
    """
        Convert measures to a numerical scale and standardize them.
    """
    def parse_measure(measure):
        numbers = [float(num) for num in measure.replace('-', ' ').split() if num.replace('.', '', 1).isdigit()]
        return np.mean(numbers) if numbers else 0

    ingredients_df['measure_numeric'] = ingredients_df['measure'].apply(parse_measure)
    scaler = StandardScaler()
    measure_scaled = scaler.fit_transform(ingredients_df[['measure_numeric']])
    return pd.DataFrame(measure_scaled, columns=['measure_scaled'])


def preprocess_glass_type(ingredients_df):
    """
        One-hot encode the glass types used in cocktails.
    """
    ohe = OneHotEncoder(sparse_output=False)
    glass_encoded = ohe.fit_transform(ingredients_df[['glass']])
    glass_df = pd.DataFrame(glass_encoded, columns=ohe.get_feature_names_out(['glass']))
    return glass_df





