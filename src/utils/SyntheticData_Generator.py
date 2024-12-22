import pandas as pd
import numpy as np

def generate_large_intrusion_dataset(samples=100000, features=20):
    """
    Generate a large synthetic dataset for intrusion detection.

    Args:
        samples (int): Number of samples in the dataset.
        features (int): Number of features per sample.

    Returns:
        pd.DataFrame: A large synthetic dataset.
    """
    # Feature names
    feature_names = [f"feature_{i}" for i in range(1, features + 1)]

    # Generate normal traffic data
    normal_data = np.random.normal(loc=0.5, scale=0.1, size=(int(0.7 * samples), features))
    normal_labels = np.zeros((normal_data.shape[0], 1))  # Label 0 for normal

    # Generate malicious traffic data
    malicious_data = np.random.normal(loc=0.8, scale=0.2, size=(int(0.3 * samples), features))
    malicious_labels = np.ones((malicious_data.shape[0], 1))  # Label 1 for malicious

    # Combine data and labels
    data = np.vstack((normal_data, malicious_data))
    labels = np.vstack((normal_labels, malicious_labels))

    # Shuffle the dataset
    shuffled_indices = np.random.permutation(data.shape[0])
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]

    # Create a DataFrame
    columns = feature_names + ["label"]
    df = pd.DataFrame(np.hstack((data, labels)), columns=columns)

    return df

# Generate the dataset
dataset = generate_large_intrusion_dataset(samples=100000, features=20)

# Save to CSV
dataset.to_csv("large_intrusion_detection_dataset.csv", index=False)
print("Dataset saved as 'large_intrusion_detection_dataset.csv'.")
