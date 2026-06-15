from sklearn.tree import DecisionTreeClassifier


def build_decision_tree_baseline(random_state=42, **kwargs):
    """Create the sklearn decision tree baseline used in reports."""
    return DecisionTreeClassifier(random_state=random_state, **kwargs)
