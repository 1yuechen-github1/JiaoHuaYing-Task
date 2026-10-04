import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score, roc_auc_score
import imageio.v2 as imageio  # 更新为 v2 来避免 DeprecationWarning
# import nibabel as nib  # 如果是 NIfTI 文件

def load_images(folder: str):
    """读取灰度图或 NIfTI 图像"""
    imgs = []
    filenames = []
    files = sorted(glob(os.path.join(folder, "*")))
    for f in files:
        if f.endswith('.db'):
            continue
        filenames.append(os.path.basename(f))  # 获取文件名
        if f.endswith(".npy"):
            img = np.load(f)
        elif f.endswith(".png"):
            img = imageio.imread(f)
        elif f.endswith(".nii") or f.endswith(".nii.gz"):
            img = nib.load(f).get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {f}")
        imgs.append(img.astype(np.uint8))
    return imgs, filenames

def dice_coefficient_per_class(y_true, y_pred, num_classes=4, smooth=1e-6):
    dice = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(np.uint8)
        y_pred_c = (y_pred == c).astype(np.uint8)
        intersection = np.sum(y_true_c * y_pred_c)
        if(np.sum(y_true_c) + np.sum(y_pred_c) == 0):  # 如果该类在 GT 和预测中都没有出现，定义 Dice 为 1
            dice.append(np.nan)  # 或者 append(1.0)，根据需求决定
        else:   
            dice.append((2. * intersection ) / (np.sum(y_true_c) + np.sum(y_pred_c) ))
    return dice

def iou_score_per_class(y_true, y_pred, num_classes=4, smooth=1e-6):
    iou = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(np.uint8)
        y_pred_c = (y_pred == c).astype(np.uint8)
        intersection = np.sum(y_true_c * y_pred_c)
        union = np.sum((y_true_c + y_pred_c) > 0)
        if union == 0:  
            iou.append(np.nan)  # 或者 append(1.0)，根据需求决定
        else:    
            iou.append((intersection + smooth) / (union + smooth))
    return iou


def calculate_auc_per_class(y_true, pred_prob, num_classes=4):
    auc_per_class = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int).flatten()
        if np.sum(y_true_c) > 0 and np.sum(y_true_c == 0) > 0:  # Ensure both positive and negative samples
            try:
                y_score_c = pred_prob[c].flatten()  # 取每个类的概率
                auc_per_class.append(roc_auc_score(y_true_c, y_score_c))
            except ValueError:
                auc_per_class.append(np.nan)  # 如果计算失败（如只有一个类别），返回nan
        else:
            auc_per_class.append(np.nan)  # 如果某类只有一个标签，无法计算AUC
    return auc_per_class


def convert_to_probability_map(gray_img, num_classes=4):
    """将灰度图转换为伪概率图"""
    prob_maps = np.zeros((num_classes, *gray_img.shape), dtype=np.float32)
    for c in range(num_classes):
        prob_maps[c] = (gray_img == c).astype(np.float32)  # 将灰度图转换为每个类别的概率图
    return prob_maps

# 路径
pred_path = r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\nnUnet2d\labelsTs\ai"
target_path = r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\nnUnet2d\labelsTs\gt"

preds, pred_filenames = load_images(pred_path)
targets, target_filenames = load_images(target_path)

assert len(preds) == len(targets), "预测和标签数量不一致"
assert pred_filenames == target_filenames, "文件名不一致"

num_classes = 4
dice_list = []
iou_list = []
acc_list = []
auc_list = []

for pred, target in zip(preds, targets):
    # 如果预测是灰度图（而非概率图）
    if pred.ndim == 2:  # 灰度图，形状为 [H, W]
        pred_label = pred
        pred_prob = convert_to_probability_map(pred, num_classes)  # 将灰度图转换为伪概率图
    else:
        pred_label = np.argmax(pred, axis=0)  # 获取预测标签
        pred_prob = pred  # 如果是概率图，直接使用
    
    # print("pred_prob:", np.unique(pred_prob))  # 查看概率图的唯一值

    # 计算 Dice, IoU, Accuracy
    dice_list.append(dice_coefficient_per_class(target, pred_label, num_classes))
    iou_list.append(iou_score_per_class(target, pred_label, num_classes))
    acc_list.append(accuracy_score(target.flatten(), pred_label.flatten()))

    # 计算 AUC
    auc_list.append(calculate_auc_per_class(target, pred_prob, num_classes))

# 转成 numpy
dice_list = np.array(dice_list)  # [num_samples, num_classes]
iou_list = np.array(iou_list)
acc_list = np.array(acc_list)
auc_list = np.array(auc_list)

# 创建 DataFrame 用于保存
data = {
    'Filename': pred_filenames,  # 用文件名替代 Sample Index
    'Dice Class 0': dice_list[:, 0],
    'Dice Class 1': dice_list[:, 1],
    'Dice Class 2': dice_list[:, 2],
    'Dice Class 3': dice_list[:, 3],
    'IoU Class 0': iou_list[:, 0],
    'IoU Class 1': iou_list[:, 1],
    'IoU Class 2': iou_list[:, 2],
    'IoU Class 3': iou_list[:, 3],
    'Accuracy': acc_list,
    'AUC Class 0': auc_list[:, 0],
    'AUC Class 1': auc_list[:, 1],
    'AUC Class 2': auc_list[:, 2],
    'AUC Class 3': auc_list[:, 3],
}


# 转换为 pandas DataFrame
df = pd.DataFrame(data)

# 计算每列的平均值
mean_data = {
    'Filename': ['Mean'],
    'Dice Class 0': np.mean(dice_list[:, 0]),
    'Dice Class 1': np.mean(dice_list[:, 1]),
    'Dice Class 2': np.mean(dice_list[:, 2]),
    'Dice Class 3': np.mean(dice_list[:, 3]),
    'IoU Class 0': np.mean(iou_list[:, 0]),
    'IoU Class 1': np.mean(iou_list[:, 1]),
    'IoU Class 2': np.mean(iou_list[:, 2]),
    'IoU Class 3': np.mean(iou_list[:, 3]),
    'Accuracy': np.mean(acc_list),
    'AUC Class 0': np.nanmean(auc_list[:, 0]),
    'AUC Class 1': np.nanmean(auc_list[:, 1]),
    'AUC Class 2': np.nanmean(auc_list[:, 2]),
    'AUC Class 3': np.nanmean(auc_list[:, 3]),
}

# 添加平均值行到 DataFrame
mean_df = pd.DataFrame(mean_data)
df = pd.concat([df, mean_df], ignore_index=True)

# 保存到 Excel
excel_path = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\nnUnet2d\labelsTs\metrics_results.xlsx'
df.to_excel(excel_path, index=False)

print(f"Metrics saved to {excel_path}")





# from PIL import Image
# import numpy as np

# img_path = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\nnunet-2d\labelTs\GT\0388-xly-Lower.png"

# img = Image.open(img_path)
# arr = np.array(img)

# values, counts = np.unique(arr, return_counts=True)

# for v, c in zip(values, counts):
#     print(f"像素值 {v}: {c} 个")