import csv
from collections import Counter
import numpy as np

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize




# -----------------------------
# 定量结果转定性标签
# -----------------------------


def save_confusion_matrix_and_tpfpfn(gt_list, ai_list, output_dir):
    """
    gt_list: GT 离散标签，如 [1, 3, 5, ...]
    ai_list: AI 离散标签，如 [1, 2, 5, ...]
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_list = np.asarray(gt_list, dtype=int)
    ai_list = np.asarray(ai_list, dtype=int)

    # 保证 1~8 类都显示；如果你实际只有部分类型，也可改为：
    # labels = sorted(set(gt_list) | set(ai_list))
    labels = list(range(1, 9))

    # 1. 混淆矩阵
    cm = confusion_matrix(gt_list, ai_list, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"GT_{label}" for label in labels],
        columns=[f"AI_{label}" for label in labels],
    )
    # cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    # cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    # 画混淆矩阵图
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    cm_png_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=300)
    plt.close()

    # 2. 各类别 TP / FP / FN / TN
    records = []

    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        records.append({
            "class": label,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "gt_count": int(cm[i, :].sum()),
            "ai_count": int(cm[:, i].sum()),
        })

    metrics_df = pd.DataFrame(records)
    metrics_csv_path = os.path.join(output_dir, "per_class_TP_FP_FN.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    print("混淆矩阵：")
    print(cm_df)

    print("\n每类 TP / FP / FN：")
    print(metrics_df.to_string(index=False))

    print("\n输出文件：")
    # print(cm_csv_path)
    print(cm_png_path)
    print(metrics_csv_path)

    return cm, metrics_df


def plot_roc_pr_curves(gt_list, ai_scores, output_dir):
    """
    只有拿到每一类的预测概率/置信度时才能使用。

    gt_list:
        [1, 2, 3, ...]，长度为 N

    ai_scores:
        numpy 数组，形状必须为 (N, 8)
        每行对应一个病例对 1~8 类的预测概率/置信度，例如：
        [
            [0.01, 0.80, 0.03, 0.01, 0.02, 0.01, 0.01, 0.11],
            ...
        ]
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_list = np.asarray(gt_list, dtype=int)
    ai_scores = np.asarray(ai_scores, dtype=float)

    labels = list(range(1, 9))

    if ai_scores.ndim != 2 or ai_scores.shape != (len(gt_list), len(labels)):
        raise ValueError(
            "ai_scores 形状必须为 (病例数量, 8)，"
            f"当前为 {ai_scores.shape}，病例数量为 {len(gt_list)}"
        )

    # GT 转 one-hot：N x 8
    gt_onehot = label_binarize(gt_list, classes=labels)

    # ROC 曲线
    plt.figure(figsize=(9, 7))

    for index, label in enumerate(labels):
        # 该类别 GT 中没有正样本或全是正样本时，ROC 不成立
        if len(np.unique(gt_onehot[:, index])) < 2:
            print(f"类别 {label} 无法绘制 ROC：GT 中正负样本不同时存在。")
            continue

        fpr, tpr, _ = roc_curve(
            gt_onehot[:, index],
            ai_scores[:, index],
        )
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"Class {label} (AUC={roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "roc_curve.png"),
        dpi=300,
    )
    plt.close()

    # PR 曲线
    plt.figure(figsize=(9, 7))

    for index, label in enumerate(labels):
        if np.sum(gt_onehot[:, index]) == 0:
            print(f"类别 {label} 无法绘制 PR：GT 中没有该类别。")
            continue

        precision, recall, _ = precision_recall_curve(
            gt_onehot[:, index],
            ai_scores[:, index],
        )
        ap = average_precision_score(
            gt_onehot[:, index],
            ai_scores[:, index],
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"Class {label} (AP={ap:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pr_curve.png"),
        dpi=300,
    )
    plt.close()

    print("已输出 ROC 和 PR 曲线。")



def label_map(label, jhy_w_len):
    """
    label: 原始类别，取值 1 / 2 / 3
    jhy_w_len: 该病例所有宽度测量值中的最小值
    """
    label = int(label)
    jhy_w_len = float(jhy_w_len)

    if label == 1:
        return 1 if jhy_w_len >= 2 else 2

    if label == 2:
        return 3 if jhy_w_len >= 2 else 4

    if label == 3:
        if jhy_w_len > 4:
            return 5
        elif jhy_w_len > 3:
            return 6
        elif jhy_w_len > 2:
            return 7
        else:
            # 包含 jhy_w_len <= 2，避免 2、3、4 等边界值漏分
            return 8

    raise ValueError(f"未知原始标签: {label}")


# -----------------------------
# CSV 读取与标签转换
# -----------------------------
def extract_original_label(filename):
    """
    保留你原本“倒数第 5 位是原始类别”的规则。
    例如 xxx_1.txt  -> 1

    若文件命名规则变了，需要只改这个函数。
    """
    if len(filename) < 5:
        raise ValueError(f"文件名过短，无法提取标签: {filename}")

    label_str = filename[-5:-4]

    if not label_str.isdigit():
        raise ValueError(
            f"无法从文件名中提取数字标签: {filename}，"
            f"当前提取结果为: {label_str}"
        )

    return int(label_str)


def get_label_dict(csv_path):
    """
    返回：
        {
            文件名: 定性标签
        }

    自动跳过表头；空值、非数值、NaN 不参与最小值计算。
    """
    label_dict = {}

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)

        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue

            filename = row[0].strip()

            # 跳过表头，例如第一列为 file
            if filename.lower() in {"file", "filename", "病例名"}:
                continue

            width_values = []
            for value in row[1:]:
                value = value.strip()

                if value == "":
                    continue

                try:
                    number = float(value)
                    if np.isfinite(number):
                        width_values.append(number)
                except ValueError:
                    print(
                        f"警告：跳过非数值宽度，"
                        f"文件={filename}, 行={row_index}, 值={value}"
                    )

            if not width_values:
                print(f"警告：文件没有有效宽度数据，跳过: {filename}")
                continue

            original_label = extract_original_label(filename)
            w_min = min(width_values) #取宽度最小值
            w_mean = np.mean(width_values)   #取宽度平均值

            qualitative_label = label_map(original_label, w_mean)
            label_dict[filename] = qualitative_label

    return label_dict


def get_aligned_label_lists(gt_path, ai_path):
    """
    根据文件名对齐 GT 与 AI，避免 CSV 行顺序不同而错配。
    """
    gt_dict = get_label_dict(gt_path)
    ai_dict = get_label_dict(ai_path)

    common_files = sorted(set(gt_dict) & set(ai_dict))
    gt_only = sorted(set(gt_dict) - set(ai_dict))
    ai_only = sorted(set(ai_dict) - set(gt_dict))

    if gt_only:
        print(f"警告：以下 {len(gt_only)} 个文件只存在于 GT，未参与评估：")
        print(gt_only)

    if ai_only:
        print(f"警告：以下 {len(ai_only)} 个文件只存在于 AI，未参与评估：")
        print(ai_only)

    if not common_files:
        raise ValueError("GT 与 AI 没有同名文件，无法评估。")

    gt_list = [gt_dict[file] for file in common_files]
    ai_list = [ai_dict[file] for file in common_files]

    return common_files, gt_list, ai_list


# -----------------------------
# 分类评估指标
# -----------------------------
def check_input(gt_list, ai_list):
    if len(gt_list) != len(ai_list):
        raise ValueError(
            f"GT 与 AI 长度不同：GT={len(gt_list)}, AI={len(ai_list)}"
        )

    if len(gt_list) == 0:
        raise ValueError("没有可评估的数据。")


def get_acc(gt_list, ai_list):
    """准确率 Accuracy"""
    check_input(gt_list, ai_list)

    gt_array = np.asarray(gt_list)
    ai_array = np.asarray(ai_list)

    return float(np.mean(gt_array == ai_array))


def get_precision(gt_list, ai_list, average="macro"):
    """
    Precision

    average:
        macro    每一类 Precision 的简单平均，推荐多分类使用
        weighted 按每类 GT 样本数加权平均
    """
    check_input(gt_list, ai_list)

    labels = sorted(set(gt_list) | set(ai_list))
    scores = []
    weights = []

    for label in labels:
        tp = sum(
            gt == label and ai == label
            for gt, ai in zip(gt_list, ai_list)
        )
        fp = sum(
            gt != label and ai == label
            for gt, ai in zip(gt_list, ai_list)
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        scores.append(precision)
        weights.append(sum(gt == label for gt in gt_list))

    if average == "macro":
        return float(np.mean(scores))

    if average == "weighted":
        return float(np.average(scores, weights=weights))

    raise ValueError("average 仅支持 'macro' 或 'weighted'")


def get_recall(gt_list, ai_list, average="macro"):
    """
    Recall / Sensitivity

    average:
        macro    每一类 Recall 的简单平均
        weighted 按每类 GT 样本数加权平均
    """
    check_input(gt_list, ai_list)

    labels = sorted(set(gt_list) | set(ai_list))
    scores = []
    weights = []

    for label in labels:
        tp = sum(
            gt == label and ai == label
            for gt, ai in zip(gt_list, ai_list)
        )
        fn = sum(
            gt == label and ai != label
            for gt, ai in zip(gt_list, ai_list)
        )

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(recall)
        weights.append(sum(gt == label for gt in gt_list))

    if average == "macro":
        return float(np.mean(scores))

    if average == "weighted":
        return float(np.average(scores, weights=weights))

    raise ValueError("average 仅支持 'macro' 或 'weighted'")


def get_f1_score(gt_list, ai_list, average="macro"):
    """
    F1-score：
    先分别计算每一类 F1，再做 macro 或 weighted 平均。
    """
    check_input(gt_list, ai_list)

    labels = sorted(set(gt_list) | set(ai_list))
    scores = []
    weights = []

    for label in labels:
        tp = sum(
            gt == label and ai == label
            for gt, ai in zip(gt_list, ai_list)
        )
        fp = sum(
            gt != label and ai == label
            for gt, ai in zip(gt_list, ai_list)
        )
        fn = sum(
            gt == label and ai != label
            for gt, ai in zip(gt_list, ai_list)
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        scores.append(f1)
        weights.append(sum(gt == label for gt in gt_list))

    if average == "macro":
        return float(np.mean(scores))

    if average == "weighted":
        return float(np.average(scores, weights=weights))

    raise ValueError("average 仅支持 'macro' 或 'weighted'")


def print_class_report(gt_list, ai_list):
    """打印每个类别的 TP、FP、FN、Precision、Recall、F1。"""
    labels = sorted(set(gt_list) | set(ai_list))

    print("\n各类别结果：")
    print("类别\tGT数\tAI数\tTP\tFP\tFN\tPrecision\tRecall\t\tF1")

    for label in labels:
        tp = sum(gt == label and ai == label for gt, ai in zip(gt_list, ai_list))
        fp = sum(gt != label and ai == label for gt, ai in zip(gt_list, ai_list))
        fn = sum(gt == label and ai != label for gt, ai in zip(gt_list, ai_list))

        gt_count = sum(gt == label for gt in gt_list)
        ai_count = sum(ai == label for ai in ai_list)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"{label}\t{gt_count}\t{ai_count}\t{tp}\t{fp}\t{fn}\t"
            f"{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}"
        )


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    gt_path = r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\spunet\Quantitative\gt\gt_jhy_w.csv"
    ai_path = r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\spunet\Quantitative\ai\ai_jhy_w.csv"
    output_dir = r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\spunet\Quantitative\output"
    files, gt_list, ai_list = get_aligned_label_lists(gt_path, ai_path)
    
    print(f"参与评估病例数：{len(files)}")
    print("GT 标签分布：", dict(sorted(Counter(gt_list).items())))
    print("AI 标签分布：", dict(sorted(Counter(ai_list).items())))

    print_class_report(gt_list, ai_list)

    print("\n总体结果：")
    print(f"Accuracy:           {get_acc(gt_list, ai_list):.4f}")
    print(f"Macro Precision:    {get_precision(gt_list, ai_list, 'macro'):.4f}")
    print(f"Macro Recall:       {get_recall(gt_list, ai_list, 'macro'):.4f}")
    print(f"Macro F1-score:     {get_f1_score(gt_list, ai_list, 'macro'):.4f}")

    print(f"Weighted Precision: {get_precision(gt_list, ai_list, 'weighted'):.4f}")
    print(f"Weighted Recall:    {get_recall(gt_list, ai_list, 'weighted'):.4f}")
    print(f"Weighted F1-score:  {get_f1_score(gt_list, ai_list, 'weighted'):.4f}")

    cm, metrics_df = save_confusion_matrix_and_tpfpfn(
        gt_list,
        ai_list,
        output_dir,
    )
