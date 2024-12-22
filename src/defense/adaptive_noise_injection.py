# adaptive_noise_injection.py
import numpy as np

def adaptive_noise_injection(input_data, epsilon=0.1):
    """
    Inject adaptive noise to defend against adversarial attacks.
    
    Args:
        input_data (np.ndarray): Input samples.
        epsilon (float): Noise magnitude.
        
    Returns:
        np.ndarray: Defended input samples.
    """
    noise = np.random.uniform(-epsilon, epsilon, input_data.shape)
    return np.clip(input_data + noise, 0, 1)