import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load data
def load_data(path):
    return pd.read_excel(path)

# Preprocess data
def preprocess_data(df, features):
    X = df[features].copy()
    X = X.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index

# Elbow plot
def plot_elbow(X_scaled):
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid()
    plt.savefig("feature_elbow_plot.png")
    plt.show()

# Evaluate clustering
def evaluate_clustering(X_scaled, labels):
    print("\nClustering Evaluation Metrics:")
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Calinski-Harabasz Index: {calinski_harabasz:.4f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')

# Visualize clusters
def visualize_clusters(df, features, cluster_col):
    sns.pairplot(df, vars=features, hue=cluster_col, palette='tab10')
    plt.suptitle('Cluster Visualization', y=1.02)
    plt.savefig("feature_cluster_visualization.png")
    plt.show()

# Cluster profiling
def cluster_profitting(df, features, clusters_col):
    for c in sorted(df[clusters_col].unique()):
        print(f"\nCluster {c}:\n")
        print(df[df[clusters_col] == c][features].describe())

# Main function
def main():
    data_path = (r'D:\Data science\02 KMeans\Sample - Superstore (1).xls')
    df = load_data(data_path)

    # Feature engineering
    df["sales - discount"] = df["Sales"] - df["Discount"]
    df['Profit Percentage'] = (df['Profit'] / df['Sales']) * 100
    df['Revenue per Quantity'] = df['Sales'] / df['Quantity']
    features = ['Sales', 'Quantity', 'Discount', 'Profit', "sales - discount", 'Profit Percentage', 'Revenue per Quantity']

    # Preprocess data
    X_scaled, valid_idx = preprocess_data(df, features)
    print('Data loaded and preprocessed.')

    # Elbow method
    print('\n--- Elbow Method ---')
    plot_elbow(X_scaled)

    # KMeans clustering
    n_clusters = 4  # Set based on elbow plot
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans_model.fit_predict(X_scaled)
    df.loc[valid_idx, 'Cluster'] = labels

    # Evaluate clustering
    evaluate_clustering(X_scaled, labels)

    # Cluster means
    print('\n--- Cluster Means ---')
    print(df.groupby('Cluster')[features].mean())

    # Cluster visualization
    print('\n--- Cluster Visualization ---')
    visualize_clusters(df.loc[valid_idx], features, 'Cluster')

    # Cluster profiling
    print('\n--- Cluster Profiling ---')
    cluster_profitting(df.loc[valid_idx], features, 'Cluster')

    # Save the trained KMeans model
    with open('Feature KMeans.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print("KMeans model saved as 'Feature KMeans.pkl'.")

if __name__ == '__main__':
    main()


