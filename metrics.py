def print_metrics(data, ingredients_df):
    """
        Print metrics about the number of unique cocktails and ingredients.
    """
    num_cocktails = data['name'].nunique()
    num_ingredients = ingredients_df['ingredient_name'].nunique()
    print(f"Number of cocktails: {num_cocktails}")
    print(f"Number of unique ingredients: {num_ingredients}")
