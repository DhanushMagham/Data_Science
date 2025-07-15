import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
import warnings
import pickle
warnings.filterwarnings("ignore")

#load data
def load_data(path):
    return pd.read_excel(path)

# preprocess data
def preprocess_data(df, features):
       # """Select features, handle missing values, and scale the data."""
    X= df[features].copy()
    X = X.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index

# Elow plot
def plot_elbow(X_scaled):
    # """Plot the elbow curve to determine the optimal number of clusters."""
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
    plt.savefig("elbow_plot.png")
    plt.show()
   

#Kmeans clustering
def kmeans(X_scaled, n_clusters):
    # """Perform KMeans clustering and return the cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans.labels


# Evaluate clustering
def evaluate_clustering(X_scaled, labels):
    # """Evaluate clustering performance using silhouette score, Calinski-Harabasz index, and Davies-Bouldin index."""
    print("\nClustering Evaluation Metrics:")
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Calinski-Harabasz Index: {calinski_harabasz:.4f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')


# visualize_clusters
def visualize_clusters(df, features, cluster_col):
    sns.pairplot(df, vars=features, hue=cluster_col, palette='tab10')
    plt.suptitle('Cluster Visualization', y=1.02)
    plt.savefig("cluster_visualization.png")
    plt.show()


#cluster_profitting
def cluster_profitting(df, features, clusters_col):
    for c in sorted(df[clusters_col].unique()):
        print(f"\nCluster {c}:\n")
        print(df[df[clusters_col] == c][features].describe())

def main():
    data_path = r'D:\Data science\02 KMeans\Sample - Superstore (1).xls'  # Update path if needed
    features = ['Sales', 'Quantity', 'Discount', 'Profit']
    df = load_data(data_path)
    X_scaled, valid_idx = preprocess_data(df, features)
    print('Data loaded and preprocessed.')

    print('\n--- Elbow Method ---')
    plot_elbow(X_scaled)

    n_clusters = 4  # Set based on elbow plot
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans_model.fit_predict(X_scaled)
    df.loc[valid_idx, 'Cluster'] = labels

    evaluate_clustering(X_scaled, labels)

    print('\n--- Cluster Means ---')
    print(df.groupby('Cluster')[features].mean())

    print('\n--- Cluster Visualization ---')
    visualize_clusters(df.loc[valid_idx], features, 'Cluster')

    print('\n--- Cluster Profiling ---')
    cluster_profitting(df.loc[valid_idx], features, 'Cluster')

if __name__ == '__main__':
    main()


# Save the model
# Save the trained KMeans model as a pickle file
with open('KMeans.pkl', 'wb') as f:
    pickle.dump(KMeans, f)
print("KMeans model saved as 'kmeans_model.pkl'.")