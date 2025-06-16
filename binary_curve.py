import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
 
# ======================== data ===============================
y = np.array([0, 0, 1, 1])   # label
pred = np.array([0.1, 0.4, 0.35, 0.8])  # probability or score
 
# ======================= metrics ============================
precision, recall, threshold = metrics.precision_recall_curve(y, pred)
print(recall)
print(precision)
print(threshold)
 
pr_auc = metrics.auc(recall, precision)  # 梯形块分割，建议使用
pr_auc0 = metrics.average_precision_score(y, pred)  # 小矩形块分割
 
print(pr_auc)
print(pr_auc0)
 
# ======================= PLoting =============================
plt.figure(1)
plt.plot(recall, precision, label=f"PR_AUC = {pr_auc:.2f}\nAP = {pr_auc0:.2f}",
         linewidth=2, linestyle='-', color='r', marker='o')
plt.fill_between(recall, y1=precision, y2=0, step=None, alpha=0.2, color='b')
plt.title("PR-Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0, 1.05])
plt.legend()
plt.show()