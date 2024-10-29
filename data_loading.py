# data_loading.py
import pandas as pd

def load_data(filepath):
    data = pd.read_json(filepath)
    ingredients_data = []

    for cocktail in data.itertuples():
        for ingredient in cocktail.ingredients:
            ingredients_data.append({
                'cocktail_name': cocktail.name,
                'ingredient_name': ingredient['name'],
                'measure': ingredient.get('measure', '').strip(),
                'glass': cocktail.glass  # Add glass type here
            })

    ingredients_df = pd.DataFrame(ingredients_data)
    return data, ingredients_df
