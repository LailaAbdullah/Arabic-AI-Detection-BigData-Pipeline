"""
Phase 5 - T5.1 & T5.2
Model Comparison + Feature Importance Visualization
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("reports/figures", exist_ok=True)

# ── T5.1: مقارنة النماذج ──
models = ["Logistic\nRegression", "Linear\nSVM", "Random\nForest"]
accuracy = [0.9670, 0.9597, 0.7980]
f1       = [0.9671, 0.9604, 0.7084]
auc      = [0.9905, 0.9833, 0.9472]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, accuracy, width, label='Accuracy', color='steelblue')
ax.bar(x,         f1,       width, label='F1-Score',  color='darkorange')
ax.bar(x + width, auc,      width, label='AUC-ROC',   color='green')

ax.set_ylabel('Score')
ax.set_title('Model Comparison - Batch Processing')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.6, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (a, f, r) in enumerate(zip(accuracy, f1, auc)):
    ax.text(i - width, a + 0.005, f'{a:.3f}', ha='center', fontsize=8)
    ax.text(i,         f + 0.005, f'{f:.3f}', ha='center', fontsize=8)
    ax.text(i + width, r + 0.005, f'{r:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=150)
print("Saved: model_comparison.png")

# ── T5.2: Feature Importance ──
features = ['f57\nTop50 Words', 'TF-IDF\nFeature 1', 'TF-IDF\nFeature 2',
            'TF-IDF\nFeature 3', 'TF-IDF\nFeature 4', 'f15\nWord Len Var',
            'f78\nPerplexity', 'f36\nAvg S/P']
importance = [0.0531, 0.0391, 0.0331, 0.0317, 0.0312, 0.0180, 0.0142, 0.0098]

colors = ['red' if 'f57' in f or 'f15' in f or 'f78' in f or 'f36' in f
          else 'steelblue' for f in features]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(features, importance, color=colors)
ax.set_xlabel('Feature Importance')
ax.set_title('Top Feature Importances (Random Forest)\nRed = Stylometric Features')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, importance):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=150)
print("Saved: feature_importance.png")

# ── TTR comparison ──
labels = ['Human', 'AI']
ttr    = [0.7585, 0.7248]
colors2 = ['#2196F3', '#FF5722']

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, ttr, color=colors2, width=0.4)
ax.set_ylabel('Type-Token Ratio (TTR)')
ax.set_title('Vocabulary Richness: Human vs AI')
ax.set_ylim(0.70, 0.78)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, ttr):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f'{val:.4f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig("reports/figures/ttr_comparison.png", dpi=150)
print("Saved: ttr_comparison.png")

# ── Feature differences ──
feat_names = ['f15\nWord Len\nVariance', 'f36\nAvg S/P', 'f57\nTop50\nWords', 'f78\nPerplexity']
human_vals = [1.439, 9.504, 6.841, 5.789]
ai_vals    = [1.184, 8.576, 10.151, 5.497]

x = np.arange(len(feat_names))
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 0.2, human_vals, 0.4, label='Human', color='#2196F3')
ax.bar(x + 0.2, ai_vals,    0.4, label='AI',    color='#FF5722')
ax.set_xticks(x)
ax.set_xticklabels(feat_names)
ax.set_title('Stylometric Features: Human vs AI')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/features_comparison.png", dpi=150)
print("Saved: features_comparison.png")

print("\nAll figures saved to reports/figures/")
