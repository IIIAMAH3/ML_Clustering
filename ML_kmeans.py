import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            x, y = map(float, line.strip().split())
            data.append([x,y])

    return np.array(data)

def parse_spiral_rile(filename):
    data = []
    true_labels = []
    with open(filename, "r") as f:
        for line in f:
            x, y, label = map(float, line.strip().split())
            data.append([x, y])
            true_labels.append(label)
    return np.array(data), np.array(true_labels)

def compute_silhouette_scores(data):
    k_range = range(2, 25)
    silhouette_scores = []
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)

        score = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True)
    plt.show()

def cluster_and_plot(data, init_method, title, n_clusters=15):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    centroids = kmeans.cluster_centers_

    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap="viridis", s=5, alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=50, marker="X", label="Centroids")
    plt.title(title)
    plt.legend()
    plt.show()


file_names = ["s1.txt", "s2.txt", "s3.txt", "s4.txt"]

for x in file_names:
    file_data = parse_file(f"C:/Users/Yaroslav/Downloads/{x}")
    cluster_and_plot(file_data, init_method="k-means++", title=f"{x} file with k-means++ init")


spiral_file, spiral_labels = parse_spiral_rile("C:/Users/Yaroslav/Downloads/spiral.txt")
spiral_kmeans = KMeans(n_clusters=3, n_init=10)
predicted_labels = spiral_kmeans.fit_predict(spiral_file)

plt.subplot(1,2,1)
plt.scatter(spiral_file[:, 0], spiral_file[:, 1], c=spiral_labels, cmap="viridis", s=10)
plt.title("True Clusters(Ground Truth)")

plt.subplot(1, 2, 2)
plt.scatter(spiral_file[:, 0], spiral_file[:, 1], c=predicted_labels, cmap="viridis", s=10)
plt.title("K-means Clusters (k=3)")
plt.show()