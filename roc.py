import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import csv
import json
import pandas as pd

# 读取内部验证数据（JSON格式）
df_in = pd.read_csv('results/inter_results.csv')
predicts = list(df_in['predicts'].values)
truths = list(df_in['truth'].values)


# 读取外部验证数据（CSV格式）
df_ex = pd.read_csv('results/exter_results.csv')
ex_predicts = list(df_ex['predicts'].values)
ex_truths = list(df_ex['truth'].values)

# 步骤1: 定义分类阈值（根据图片信息，使用45%作为阈值）
clinical_threshold = 45

# 步骤2: 将连续值转换为二分类标签 - 内部验证数据
y_true_binary_int = np.array([1 if ef < clinical_threshold else 0 for ef in truths])
distance_from_threshold_int = clinical_threshold - np.array(predicts)
y_pred_prob_int = 1 / (1 + np.exp(-distance_from_threshold_int / 5))
y_pred_prob_int = np.clip(y_pred_prob_int, 0, 1)

# 步骤2: 将连续值转换为二分类标签 - 外部验证数据
y_true_binary_ext = np.array([1 if ef < clinical_threshold else 0 for ef in ex_truths])
distance_from_threshold_ext = clinical_threshold - np.array(ex_predicts)
y_pred_prob_ext = 1 / (1 + np.exp(-distance_from_threshold_ext / 5))
y_pred_prob_ext = np.clip(y_pred_prob_ext, 0, 1)

# 步骤3: 计算ROC曲线
fpr_int, tpr_int, _ = roc_curve(y_true_binary_int, y_pred_prob_int)
fpr_ext, tpr_ext, _ = roc_curve(y_true_binary_ext, y_pred_prob_ext)

# 计算AUC值
auc_int = auc(fpr_int, tpr_int)
auc_ext = auc(fpr_ext, tpr_ext)

# 步骤4: 使用Bootstrap计算置信区间
n_bootstraps = 10000  # 根据图片信息，使用10000次bootstrap
rng_seed = 42

# 内部验证数据的Bootstrap
bootstrapped_auc_int = []
tprs_int = []
base_fpr = np.linspace(0, 1, 101)

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # 内部验证bootstrap
    indices = rng.randint(0, len(y_true_binary_int), len(y_true_binary_int))
    if len(np.unique(y_true_binary_int[indices])) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(y_true_binary_int[indices], y_pred_prob_int[indices])
    auc_b = auc(fpr_b, tpr_b)
    bootstrapped_auc_int.append(auc_b)
    interp_tpr = np.interp(base_fpr, fpr_b, tpr_b)
    interp_tpr[0] = 0.0
    tprs_int.append(interp_tpr)

# 外部验证数据的Bootstrap
bootstrapped_auc_ext = []
tprs_ext = []

for i in range(n_bootstraps):
    # 外部验证bootstrap
    indices = rng.randint(0, len(y_true_binary_ext), len(y_true_binary_ext))
    if len(np.unique(y_true_binary_ext[indices])) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(y_true_binary_ext[indices], y_pred_prob_ext[indices])
    auc_b = auc(fpr_b, tpr_b)
    bootstrapped_auc_ext.append(auc_b)
    interp_tpr = np.interp(base_fpr, fpr_b, tpr_b)
    interp_tpr[0] = 0.0
    tprs_ext.append(interp_tpr)

# 计算AUC置信区间
auc_int_ci_lower = np.percentile(bootstrapped_auc_int, 2.5)
auc_int_ci_upper = np.percentile(bootstrapped_auc_int, 97.5)
auc_ext_ci_lower = np.percentile(bootstrapped_auc_ext, 2.5)
auc_ext_ci_upper = np.percentile(bootstrapped_auc_ext, 97.5)

# 计算TPR置信区间
tprs_int = np.array(tprs_int)
tprs_ext = np.array(tprs_ext)
tpr_int_lower = np.percentile(tprs_int, 2.5, axis=0)
tpr_int_upper = np.percentile(tprs_int, 97.5, axis=0)
tpr_ext_lower = np.percentile(tprs_ext, 2.5, axis=0)
tpr_ext_upper = np.percentile(tprs_ext, 97.5, axis=0)

# 步骤5: 创建图表并绘制ROC曲线
# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制内部验证ROC曲线和置信区间
ax.plot(fpr_int, tpr_int, color='blue', lw=2, 
         label=f'Internal validation (AUC = {auc_int:.3f} [95% CI: {auc_int_ci_lower:.3f}-{auc_int_ci_upper:.3f}])')
ax.fill_between(base_fpr, tpr_int_lower, tpr_int_upper, color='blue', alpha=0.2)

# 绘制外部验证ROC曲线和置信区间
ax.plot(fpr_ext, tpr_ext, color='red', lw=2, 
         label=f'External validation (AUC = {auc_ext:.3f} [95% CI: {auc_ext_ci_lower:.3f}-{auc_ext_ci_upper:.3f}])')
ax.fill_between(base_fpr, tpr_ext_lower, tpr_ext_upper, color='red', alpha=0.2)

# 添加对角线参考线
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random classifier (AUC = 0.5)')

# 设置图表属性
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=14)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=14)
ax.set_title('ROC Curves for RVEF-based Right Ventricular Dysfunction Detection', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)

# 添加样本量信息
plt.figtext(0.5, 0.02, 
           f'Internal validation: n = {len(truths)}, External validation: n = {len(ex_truths)}\n'
           f'Classification threshold: RVEF < {clinical_threshold}%\n'
           f'95% confidence intervals calculated with {n_bootstraps} bootstrap resamples',
           ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('save_fig/ROC_Curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印统计信息
print("=== INTERNAL VALIDATION ===")
print(f"样本数量: {len(truths)}")
print(f"真实阳性数 (RVEF < {clinical_threshold}%): {sum(y_true_binary_int)}")
print(f"真实阴性数 (RVEF ≥ {clinical_threshold}%): {len(y_true_binary_int) - sum(y_true_binary_int)}")
print(f"AUC: {auc_int:.3f} (95% CI: {auc_int_ci_lower:.3f}-{auc_int_ci_upper:.3f})")

print("\n=== EXTERNAL VALIDATION ===")
print(f"样本数量: {len(ex_truths)}")
print(f"真实阳性数 (RVEF < {clinical_threshold}%): {sum(y_true_binary_ext)}")
print(f"真实阴性数 (RVEF ≥ {clinical_threshold}%): {len(y_true_binary_ext) - sum(y_true_binary_ext)}")
print(f"AUC: {auc_ext:.3f} (95% CI: {auc_ext_ci_lower:.3f}-{auc_ext_ci_upper:.3f})")