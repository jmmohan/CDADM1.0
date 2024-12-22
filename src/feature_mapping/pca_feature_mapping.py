# pca_feature_mapping.py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def map_features(data, n_components=10):
    """
    Map domain-specific features to a universal feature space using PCA.
    
    Args:
        data (np.ndarray): Input feature data.
        n_components (int): Number of components for dimensionality reduction.
        
    Returns:
        np.ndarray: Universal feature representation.
    """
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    return pca.fit_transform(normalized_data)