from sklearn.ensemble import HistGradientBoostingClassifier

THRESHOLD = 0.251797

def create_model():
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        min_samples_leaf=81, 
        max_depth=3
    )
    return model
