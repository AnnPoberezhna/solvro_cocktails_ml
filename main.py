from sklearn.decomposition import PCA
from data_loading import load_data
from preprocessing import preprocess_ingredients, preprocess_measures, preprocess_glass_type
from clustering import apply_dbscan, calculate_silhouette_score
from visualization import plot_clusters
from metrics import print_metrics
import pandas as pd
from eda import perform_eda

data, ingredients_df = load_data("cocktail_dataset.json")

# EDA
perform_eda(data)

# Preprocess ingredients, measures, and glass type
ingredient_names_df = preprocess_ingredients(ingredients_df)
measure_scaled_df = preprocess_measures(ingredients_df)
glass_df = preprocess_glass_type(ingredients_df)

final_data = pd.concat([measure_scaled_df, ingredient_names_df, glass_df], axis=1)

# Reduce to 2 dimensions for better visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(final_data)

# Apply DBSCAN clustering
dbscan_clusters = apply_dbscan(reduced_data, eps=0.5, min_samples=5)

# Calculate Silhouette Score for DBSCAN Clusters
dbscan_silhouette_score = calculate_silhouette_score(reduced_data, dbscan_clusters)
if dbscan_silhouette_score is not None:
    print(f"Silhouette Score for DBSCAN clustering: {dbscan_silhouette_score}")
else:
    print("Only one cluster found; silhouette score is not applicable.")


plot_clusters(ingredients_df, reduced_data, dbscan_clusters)

# Display metrics
print_metrics(data, ingredients_df)

print("The final visualization can be found in the /result folder.")




