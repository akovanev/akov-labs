import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
# Rows: Actual [Negative, Positive]
# Columns: Predicted [Negative, Positive]
# TN=50, FP=10, FN=0, TP=40
cm = np.array([
    [50, 10],
    [0, 40]
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: Standard confusion matrix ---
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax1,
    xticklabels=['Predicted Negative', 'Predicted Positive'],
    yticklabels=['Actual Negative', 'Actual Positive']
)
ax1.set_title('Confusion Matrix\n(TN=50, FP=10, FN=0, TP=40)')

# --- Plot 2: Correct predictions only ---
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

correct = cm.copy()
correct[0, 1] = 0  # FP masked
correct[1, 0] = 0  # FN masked

pred_neg_total = cm[:, 0].sum()  # 50
pred_pos_total = cm[:, 1].sum()  # 50

sns.heatmap(
    correct,
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=ax2,
    xticklabels=[
        f'Pred Neg ({cm[0,0]}/{pred_neg_total})',
        f'Pred Pos ({cm[1,1]}/{pred_pos_total})'
    ],
    yticklabels=['Actual Negative', 'Actual Positive']
)

ax2.set_title(
    f'Correct Predictions (Green)\n'
    f'Accuracy = {accuracy:.2%} ({cm[0,0] + cm[1,1]}/{cm.sum()})'
)

plt.tight_layout()
plt.show()
