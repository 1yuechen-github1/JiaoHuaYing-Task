import os
from pathlib import Path

import numpy as np
import open3d as o3d

from utils import (
    keep_largest_connected_component,
    nearest_distance_mask,
    rotate_point_cloud,
)

DISTANCE_THRESHOLD = 0.1
CONNECT_RADIUS = 0.60
MIN_COMPONENT_POINTS = 100

KERATINIZED_GUMS_ROOT = (
    r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\带线分割测试(1)\带线分割测试\piontcloud"
)
KERATINIZED_GUMS_GT_DIR = os.path.join(KERATINIZED_GUMS_ROOT, "gt")
KERATINIZED_GUMS_AI_DIR = os.path.join(KERATINIZED_GUMS_ROOT, "ai")
MISSING_TEETH_AREA_DIR = (
    r"Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\cropped_debug"
)
OUTPUT_DIR = os.path.join(KERATINIZED_GUMS_ROOT, "merge")
DATA_LIST = ["0002", "0004"]


def is_valid_gum_file(file_name):
    name = file_name.lower()
    return name.endswith(".ply") and "upp" not in name and "low" not in name


def load_gum_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(str(file_path))
    points = np.asarray(pcd.points, dtype=np.float64)

    if len(points) == 0:
        raise ValueError(f"点云为空：{file_path}")

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float64)
    else:
        colors = np.ones((len(points), 3), dtype=np.float64)

    if len(colors) != len(points):
        colors = np.ones((len(points), 3), dtype=np.float64)

    if colors.max() <= 1.0:
        colors *= 255.0

    colors = np.clip(np.rint(colors), 0, 255).astype(np.int32)
    return np.hstack((points, colors))


def rotate_gum_point_cloud(data, file_name):
    return rotate_point_cloud(data, axis="x", angle_deg=0)


def get_gum_files(gum_dir, patient_id):
    patient_dir = os.path.join(gum_dir, patient_id)

    if not os.path.isdir(patient_dir):
        return []

    return sorted(
        os.path.join(patient_dir, file_name)
        for file_name in os.listdir(patient_dir)
        if is_valid_gum_file(file_name)
    )


def get_missing_area_files(patient_id):
    files = []

    for root, _, file_names in os.walk(MISSING_TEETH_AREA_DIR):
        files.extend(
            os.path.join(root, file_name)
            for file_name in file_names
            if file_name.lower().endswith(".txt")
            and file_name.startswith(patient_id)
        )

    return sorted(files)


def get_label_from_file_name(file_name):
    try:
        return int(Path(file_name).name[-5:-4])
    except ValueError as exc:
        raise ValueError(
            f"无法从文件名提取标签：{file_name}"
        ) from exc


def load_missing_area(file_path):
    data = np.asarray(np.loadtxt(file_path), dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(f"缺牙区文件至少需要 3 列 XYZ：{file_path}")

    xyz = data[:, :3]

    if data.shape[1] >= 6:
        rgb = data[:, 3:6]
    else:
        rgb = np.full((len(xyz), 3), 255, dtype=np.float64)

    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int32)
    return np.hstack((xyz, rgb))


def calculate_label(missing_area, gum_points, label):
    candidate_mask, _ = nearest_distance_mask(
        missing_area[:, :3],
        gum_points[:, :3],
        distance_threshold=DISTANCE_THRESHOLD,
    )

    mask, component_count, largest_component_size = (
        keep_largest_connected_component(
            missing_area[:, :3],
            candidate_mask,
            connect_radius=CONNECT_RADIUS,
            min_component_points=MIN_COMPONENT_POINTS,
        )
    )

    labels = np.zeros(len(missing_area), dtype=np.int32)
    labels[mask] = label

    return (
        labels,
        component_count,
        largest_component_size,
        int(np.count_nonzero(labels > 0)),
    )


def make_gum_map(gum_files):
    if not gum_files:
        return None

    all_data = []

    for gum_file in gum_files:
        print(f"读取角化牙龈：{gum_file}")

        data = load_gum_point_cloud(gum_file)
        data = rotate_gum_point_cloud(data, Path(gum_file).name)
        all_data.append(data)

    return np.vstack(all_data)


def choose_best_missing_area(missing_files, gum_points, label_name):
    candidates = []

    for missing_file in missing_files:
        print(f"\n处理 {label_name}：{Path(missing_file).name}")

        missing_area = load_missing_area(missing_file)
        label = get_label_from_file_name(missing_file)

        (
            labels,
            component_count,
            largest_component_size,
            positive_count,
        ) = calculate_label(missing_area, gum_points, label)

        candidates.append(
            {
                "file": missing_file,
                "points": missing_area,
                "labels": labels,
                "positive_count": positive_count,
            }
        )

        print(
            f"{label_name} 正样本点数：{positive_count}，"
            f"连通区域数：{component_count}，"
            f"最大连通区域：{largest_component_size}"
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["positive_count"])


def save_result(output_file, points, gt_labels, ai_labels):
    output = np.column_stack(
        (
            points[:, :3],
            points[:, 3:6],
            gt_labels,
            ai_labels,
        )
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_file,
        output,
        fmt=["%.6f"] * 3 + ["%d"] * 5,
        # header="x y z r g b gt ai",
        comments="",
    )

    print(f"\n保存完成：{output_file}")
    print(f"总点数：{len(output)}")
    print(f"GT 正样本数：{np.count_nonzero(gt_labels > 0)}")
    print(f"AI 正样本数：{np.count_nonzero(ai_labels > 0)}")


def process_patient(patient_id):
    print(f"\n{'=' * 60}")
    print(f"开始处理患者：{patient_id}")

    gt_files = get_gum_files(KERATINIZED_GUMS_GT_DIR, patient_id)
    ai_files = get_gum_files(KERATINIZED_GUMS_AI_DIR, patient_id)
    missing_files = get_missing_area_files(patient_id)

    print(f"GT 角化牙龈文件数：{len(gt_files)}")
    print(f"AI 角化牙龈文件数：{len(ai_files)}")
    print(f"缺牙区文件数：{len(missing_files)}")

    required_files = (
        (gt_files, "GT 角化牙龈文件"),
        (ai_files, "AI 角化牙龈文件"),
        (missing_files, "缺牙区文件"),
    )

    for files, description in required_files:
        if not files:
            print(f"{patient_id} 没有找到{description}")
            return

    best_gt = choose_best_missing_area(
        missing_files,
        make_gum_map(gt_files),
        "GT",
    )

    best_ai = choose_best_missing_area(
        missing_files,
        make_gum_map(ai_files),
        "AI",
    )

    selected = best_gt or best_ai

    if selected is None:
        print(f"{patient_id} 没有可用结果")
        return

    selected_file_name = Path(selected["file"]).name
    points = selected["points"]

    gt_labels = np.zeros(len(points), dtype=np.int32)
    ai_labels = np.zeros(len(points), dtype=np.int32)

    if best_gt and Path(best_gt["file"]).name == selected_file_name:
        gt_labels = best_gt["labels"]

    if best_ai and Path(best_ai["file"]).name == selected_file_name:
        ai_labels = best_ai["labels"]

    output_file = os.path.join(OUTPUT_DIR, selected_file_name)

    if os.path.exists(output_file):
        print(f"文件已经存在，跳过：{output_file}")
        return

    save_result(output_file, points, gt_labels, ai_labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for patient_id in DATA_LIST:
        process_patient(patient_id)


if __name__ == "__main__":
    main()