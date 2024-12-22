# Cross-Domain Adaptive Defense Mechanism (CDADM)

This repository contains the implementation of the Cross-Domain Adaptive Defense Mechanism (CDADM) for detecting and mitigating adversarial attacks across multiple domains, including Computer Vision, Natural Language Processing (NLP), and Cybersecurity.
The Cross-Domain Adaptive Defense Mechanism (CDADM) is a framework designed to enhance the robustness of machine learning models against adversarial attacks across different domains. The mechanism adapts to various data types (e.g., images, text, or network traffic) while offering real-time defenses against adversarial perturbations. The system integrates multiple techniques, including Adaptive Defense Layers (ADL), Universal Perturbation Detection (UPD), and Cross-Modal Feature Mapping (CMFM), to protect models from attacks across diverse domains.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Datasets](#datasets)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Overview**
Adversarial attacks exploit vulnerabilities in machine learning models by perturbing input data to produce incorrect predictions. The CDADM framework identifies shared adversarial patterns across domains and mitigates their effects through:
- Universal perturbation detection.
- Cross-modal feature mapping.
- Adaptive defense adjustment.

---

## **Features**
- Detection of adversarial attacks across diverse domains.
- Defense through adaptive noise injection.
- Compatibility with image, text, and tabular data models.
- Scalable and domain-agnostic feature mapping using PCA and neural networks.

---

## **Requirements**
Refer requirements.txt file for the project, listing all the dependencies needed for the Python scripts in the GitHub structure

```plaintext
# General Dependencies
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2

# PyTorch for deep learning models
torch==2.0.1
torchvision==0.15.2

# Transformers for BERT-based models
transformers==4.30.2

# ART (Adversarial Robustness Toolbox) for adversarial sample generation
adversarial-robustness-toolbox==1.13.0

# Utilities
matplotlib==3.7.2
seaborn==0.12.2
pickle-mixin==1.0.2
```

### Explanation of Packages
1. **`numpy` and `pandas`**:
   - For numerical computations and tabular data manipulation.
2. **`scikit-learn`**:
   - For tabular data modeling (e.g., `RandomForestClassifier`) and preprocessing (e.g., PCA).
3. **`torch` and `torchvision`**:
   - For building and using deep learning models like ResNet and custom domain encoders.
4. **`transformers`**:
   - To load pre-trained BERT models for text classification.
5. **`adversarial-robustness-toolbox`**:
   - For generating adversarial samples using methods like FGSM.
6. **`matplotlib` and `seaborn`**:
   - For visualization of metrics and evaluation results.
7. **`pickle-mixin`**:
   - To load and save models/datasets efficiently.

### Commands to Install
To install all dependencies at once, use the following command in your project directory:

```bash
pip install -r requirements.txt
```


## **Installation**
To set up the repository locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jmmohan/CDADM1.0
   cd cdadm1.0


## **Usage**

### 1. Generate Adversarial Samples
```python
from adversarial_attack_generation import generate_adversarial_samples
X_adv = generate_adversarial_samples(model, X_test, y_test)
```

### 2. Detect Adversarial Samples
```python
from adversarial_detection import detect_adversarial
flags = detect_adversarial(X_adv, model)
```

### 3. Apply CDADM Framework
```python
from cdadm_framework import cdadm_framework
results = cdadm_framework(X_adv, model)
print(results)
```

---

## **Project Structure**
```
├── adversarial_detection.py        # Detect adversarial samples
├── gradient_norms.py               # Compute gradient norms
├── pca_feature_mapping.py          # Feature mapping using PCA
├── domain_encoder.py               # Neural network for feature mapping
├── adaptive_noise_injection.py     # Inject adaptive noise
├── cdadm_framework.py              # Main CDADM framework implementation
├── image_model.py                  # Image classification model
├── text_model.py                   # NLP model with BERT
├── tabular_model.py                # Random forest for tabular data
├── baseline_evaluation.py          # Baseline evaluation scripts
├── adversarial_attack_generation.py# Generate adversarial samples
├── cdadm_evaluation.py             # Evaluate CDADM mechanism
├── utils.py                        # Utility functions for data/model loading
├── requirements.txt                # Dependency list
```

---

## **Datasets**
- **Computer Vision**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **NLP**: [IMDb Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Cybersecurity**: Synthetic tabular data for intrusion detection. Use the SyntheticData_Generator.py to generate tabular data for testing.  Find the sample data in Src/Data folder.  

---

## **Results**
| Metric                  | Baseline     | CDADM        |
|-------------------------|--------------|--------------|
| Accuracy (Clean Data)   | 85%          | 90%          |
| Accuracy (Adversarial Data) | 60%      | 85%          |
| True Positive Rate (TPR)| 65%          | 92%          |
| False Positive Rate (FPR)| 15%         | 8%           |
| Runtime Overhead (sec)  | 0.05         | 0.12         |

## **Interpretation : **
**Accuracy:**
On clean datasets, CDADM maintains high accuracy across domains.
On adversarial datasets, CDADM significantly outperforms the baseline by 25%.

**Detection Rates:**
CDADM exhibits a much higher TPR (92%) compared to the baseline (65%), effectively detecting adversarial samples.
The FPR of CDADM is lower, ensuring fewer false alarms.

**Runtime Overhead:**
While CDADM introduces a slight increase in computational cost (from 0.05s to 0.12s per sample), the improved robustness justifies the trade-off.
Cross-Domain Performance:

**CDADM demonstrates excellent generalization across diverse data modalities (image, text, tabular), unlike the baseline, which struggles with unseen domains.**

---

## **Contributing**
Contributions are welcome! Please fork the repository and create a pull request.

---

## **License**
This project is licensed under the MIT License.
```


