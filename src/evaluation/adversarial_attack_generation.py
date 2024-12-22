# adversarial_attack_generation.py
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

def generate_adversarial_samples(model, X, y):
    """
    Generate adversarial samples using FGSM.
    
    
    Args:
        model (object): Trained ML model.
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        
    Returns:
        np.ndarray: Adversarial samples.
    """
    classifier = SklearnClassifier(model=model)
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    return attack.generate(X)