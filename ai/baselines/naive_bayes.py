from sklearn.naive_bayes import GaussianNB


def build_naive_bayes_baseline(**kwargs):
    """Create the sklearn Gaussian Naive Bayes baseline used in reports."""
    return GaussianNB(**kwargs)
