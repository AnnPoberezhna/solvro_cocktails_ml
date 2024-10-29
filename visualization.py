import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_clusters(ingredients_df, data, clusters):
    """
        Visualize the clusters using t-SNE dimensionality reduction.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)
    ingredients_df['tsne_x'] = tsne_results[:, 0]
    ingredients_df['tsne_y'] = tsne_results[:, 1]
    ingredients_df['cluster'] = clusters

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        ingredients_df['tsne_x'], ingredients_df['tsne_y'],
        c=ingredients_df['cluster'], cmap='viridis', s=50, alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.title("Visualization of cocktail clusters using t-SNE.")
    plt.savefig('./result/result_dbscan.png')
    plt.show()
