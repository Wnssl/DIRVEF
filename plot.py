import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

def font_setting():
    font_path = 'Times New Roman.ttf'
    fontManager.addfont(font_path)


def calculate_metrics(y_true, y_pred):
    """
    calculate regression metrics：MAE, MSE, R²

    para:
    y_true -- ground_truth
    y_pred -- predicts

    return:
    (mae, mse, r2)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("输入数组长度不一致")
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


def plot_scatter(filename,save):
    """
    filename : input of predicts file type(csv)
    save : save_path
    """
    font_setting()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  

   
    df = pd.read_csv(filename)
    ground_truth = df['truth'].values
    predicted = df['predicts'].values
    
    slope, intercept, r_value, _, _ = stats.linregress(ground_truth, predicted)
    x_range = np.array([20, 75])
    reg_line = slope * x_range + intercept

    
    s_x = np.linspace(20, 75, 50)
    s_y = slope * s_x + intercept
    residuals = predicted - (slope * ground_truth + intercept)
    stderr = np.sqrt(np.sum(residuals**2) / (len(ground_truth) - 2))
    mean_x = np.mean(ground_truth)
    conf_interval = stderr * np.sqrt(1/len(ground_truth) + (s_x - mean_x)**2 / np.sum((ground_truth - mean_x)**2))
    conf_interval *= stats.t.ppf(0.975, len(ground_truth) - 2)

    
    plt.figure(figsize=(10, 8), facecolor='white')
    plt.grid(False)

    
    plt.scatter(ground_truth, predicted, color='#1F77B4', alpha=0.7, s=50,  # 修改点：s=15 -> s=25
                edgecolor='white', linewidth=0.5)


    plt.plot(s_x, s_y - conf_interval, '--', color='#FFA07A', alpha=0.8, linewidth=1.2)
    plt.plot(s_x, s_y + conf_interval, '--', color='#FFA07A', alpha=0.8, linewidth=1.2)
    plt.plot(x_range, reg_line, color='#FF7F50', linewidth=2.5)
    plt.axvline(x=45, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.axhline(y=45, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)

    plt.xlim(20, 75)
    plt.ylim(20, 75)
    plt.xticks(np.arange(20, 76, 10))
    plt.yticks(np.arange(20, 76, 10))

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xlabel('Ground Truth RVEF (%)', fontsize=16)
    plt.ylabel('DL Predicted RVEF (%)', fontsize=16)
    _,_,r2 = calculate_metrics(ground_truth, predicted)
    
    plt.text(21, 71, f'$R^2$ ={round(r2,2)}', fontsize=20, weight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.show()
    plt.close()

    print(f"figure saved in :{save}")

def bland_altman_plot(predicted, actual, save_path=None):
    """
    Draw a refined Bland-Altman plot with optimized layout
    
    Parameters:
    predicted -- Array of predicted values
    actual -- Array of actual values
    save_path -- Path to save the image (optional)
    """
    # Set up clean style
    sns.set_style("white")
    plt.figure(figsize=(10, 7), dpi=150)
    
    # Calculate differences and means
    diff = predicted - actual
    mean = np.mean([predicted, actual], axis=0)
    
    # Compute key statistics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa = 1.96 * std_diff
    
    # Create optimized color palette
    primary_color = "#1F77B4"   # Professional blue
    secondary_color = "#2E5984" # Slightly darker for lines
    fill_color = "#87CEEB"      # Light blue for fill (80% transparency)
    background_color = "#FFFFFF"
    
    # Set background color
    plt.gca().set_facecolor(background_color)
    
    # Create scatter plot with improved settings
    plt.scatter(mean, diff, 
                alpha=0.75, 
                c=primary_color,
                edgecolors="none",  # Remove any edges for pure color
                s=45,               # Slightly smaller points
                zorder=5,
                label='Data points')
    
    # Add limits of agreement lines
    plt.axhline(mean_diff + loa, color=secondary_color, 
                linestyle='--', linewidth=1.8, 
                zorder=3, label=f'±1.96 SD ({mean_diff + loa:.2f} to {mean_diff - loa:.2f})')
    plt.axhline(mean_diff - loa, color=secondary_color, 
                linestyle='--', linewidth=1.8, 
                zorder=3)
    
    # Add LoA shading with transparency
    plt.fill_between([min(mean), max(mean)], 
                     [mean_diff - loa, mean_diff - loa],
                     [mean_diff + loa, mean_diff + loa],
                     color=fill_color, alpha=0.2, zorder=2,
                     label='95% Agreement Interval')
    
    # Add title and labels (remove duplicate labels)
    plt.title('Bland-Altman Analysis', fontsize=18, pad=12, weight='bold')
    plt.xlabel('Mean of Measurements (Predicted + Actual)/2', fontsize=16, labelpad=8)
    plt.ylabel('Difference (Predicted - Actual)', fontsize=16, labelpad=8)
    
    # Calculate regression line for potential trend
    slope, intercept, r_value, p_value, _ = stats.linregress(mean, diff)
    reg_line = slope * mean + intercept
    
    # Add regression line if slope is significant (use absolute slope for threshold)
    if abs(slope) > 1e-3:  
        plt.plot(mean, reg_line, '-', color='#333333', linewidth=1.8, 
                 label=f'Trend (p={p_value:.1e})')
    
    # Set axis limits with padding
    x_range = max(mean) - min(mean)
    y_range = max(diff) - min(diff)
    y_padding = 0.2 * y_range  # Increase vertical padding
    
    plt.xlim(min(mean) - 0.06*x_range, max(mean) + 0.12*x_range)  # More space on right for labels
    plt.ylim(mean_diff - 3.7*std_diff, mean_diff + 3.7*std_diff)
    
    # Calculate offset to prevent label-line overlap
    label_offset_y = 0.08 * (max(diff) - min(diff))
    label_offset_x = 0.03 * x_range
    
    # ============ 调整公式位置 ============
    # 将公式向左移动，避免与图例重叠
    formula_offset_x = 0.08 * x_range  # 向左移动的距离
    
    # 调整 +1.96 SD 标签位置（向左移动，稍微降低高度）
    plt.text(max(mean) + formula_offset_x, mean_diff + loa - label_offset_y, 
             f'+1.96 SD = {mean_diff + loa:.2f}', 
             color=secondary_color, fontsize=14,
             verticalalignment='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                     edgecolor='none'))
    
    # 调整 -1.96 SD 标签位置（向左移动，稍微提高高度）
    plt.text(max(mean) + formula_offset_x, mean_diff - loa + label_offset_y, 
             f'-1.96 SD = {mean_diff - loa:.2f}', 
             color=secondary_color, fontsize=14,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, 
                     edgecolor='none'))
    
    # 调整均值标签位置（向左移动，居中显示）
    plt.text(max(mean) + formula_offset_x, mean_diff, 
             f'Mean = {mean_diff:.2f}', 
             color='#444444', fontsize=14, fontweight='bold',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                     edgecolor='none'))
    
    # ============ 保持图例位置不变 ============
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 保持图例在右上角（不改变位置）
    legend = plt.legend(handles, labels, 
               loc='upper right', 
               fontsize=12, 
               frameon=True, 
               framealpha=0.85,
               facecolor='white', 
               edgecolor='#CCCCCC',
               bbox_to_anchor=(1, 1), 
               borderaxespad=0.)
    
    # 添加参考零线
    plt.axhline(0, color='#888888', linestyle='-', linewidth=1.0, alpha=0.8, zorder=1)
    
    # 美化坐标轴
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.15)
    sns.despine(left=True, bottom=True)
    plt.gca().spines['bottom'].set_color('#555555')
    plt.gca().spines['left'].set_color('#555555')
    
    # 调整布局确保所有元素可见
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Add right margin for labels
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=180)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
if __name__ == '__main__':
    plot_scatter("results/exter_results.csv", "save_fig/external_scatter.png")
    plot_scatter("results/inter_results.csv", "save_fig/internal_scatter.png")
    np.random.seed(42)
    df = pd.read_csv("results/exter_results.csv")
    predicts = df['predicts'].values
    truths = df['truth'].values
    bland_altman_plot(np.array(predicts), np.array(truths),"save_fig/bland_altman_external.png")
    
    df = pd.read_csv("results/inter_results.csv")
    predicts = df['predicts'].values
    truths = df['truth'].values
    bland_altman_plot(np.array(predicts), np.array(truths),"save_fig/bland_altman_internal.png")