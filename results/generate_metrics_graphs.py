import matplotlib.pyplot as plt
import numpy as np

# Sample metrics for illustration
baseline_metrics = {
    "accuracy": 0.82,
    "precision": 0.79,
    "recall": 0.76,
    "f1_score": 0.77,
}

cdadm_metrics = {
    "accuracy": 0.91,
    "precision": 0.88,
    "recall": 0.85,
    "f1_score": 0.86,
}

# Combine metrics
metrics_names = list(baseline_metrics.keys())
baseline_values = list(baseline_metrics.values())
cdadm_values = list(cdadm_metrics.values())

# Plotting bar chart for metrics
x = np.arange(len(metrics_names))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline')
bars2 = ax.bar(x + width/2, cdadm_values, width, label='CDADM')

# Add text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics: Baseline vs CDADM')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Add value annotations on the bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

add_values(bars1)
add_values(bars2)

fig.tight_layout()
plt.show()
