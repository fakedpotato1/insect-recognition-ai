from sklearn.neighbors import KNeighborsClassifier


def build_knn_baseline(n_neighbors=5, **kwargs):
    """Create the sklearn KNN baseline used in reports."""
    return KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
