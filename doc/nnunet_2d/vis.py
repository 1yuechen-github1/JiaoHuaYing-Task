import os
from pathlib import Path
import cv2
import numpy as np


MIN_CONTOUR_AREA = 5000
GREEN = (0, 180, 0)
RED = (0, 0, 255)
LINE_WIDTH = 3


# def get_mask_polygons(mask, min_area=MIN_CONTOUR_AREA):
#     """从二值 Mask 中提取四边形；无法拟合时使用最小旋转矩形。"""
#     binary_mask = (mask > 0).astype(np.uint8)

#     contours, _ = cv2.findContours(
#         binary_mask,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE,
#     )

#     polygons = []

#     for contour in contours:
#         if cv2.contourArea(contour) < min_area:
#             continue

#         perimeter = cv2.arcLength(contour, True)
#         quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

#         if len(quad) == 4:
#             polygon = quad.astype(np.int32)
#         else:
#             rect = cv2.minAreaRect(contour)
#             polygon = np.round(cv2.boxPoints(rect)).astype(np.int32)

#         polygons.append(polygon)

#     return polygons

def get_mask_polygons(mask, min_area=MIN_CONTOUR_AREA):
    binary_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # 约 1%：轮廓更贴合 mask；数值越大，轮廓越简单
        epsilon = 0.01 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        polygons.append(polygon)

    return polygons


def make_binary_mask(shape, polygons):
    """生成四边形内部为白色、外部为黑色的图像。"""
    output = np.zeros(shape[:2], dtype=np.uint8)

    if polygons:
        cv2.fillPoly(output, polygons, 255)

    return output


def overlay_mask(image, mask, color, alpha=0.35):
    """在图像上半透明覆盖指定颜色。"""
    output = image.copy()
    color_layer = image.copy()
    color_layer[mask > 0] = color

    blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
    output[mask > 0] = blended[mask > 0]
    return output


def frame_mask(seg_img, gt_img, ori_img, output_dir, filename):
    polygons = get_mask_polygons(seg_img)

    # _2：AI 四边形区域黑白图
    ai_box_mask = make_binary_mask(ori_img.shape, polygons)

    # _3：原图 + AI 绿色填充 + GT 红色半透明填充
    filled_result = overlay_mask(ori_img, ai_box_mask, GREEN, alpha=0.35)
    filled_result = overlay_mask(filled_result, gt_img, RED, alpha=0.35)

    # _4：原图 + AI 绿色四边形框 + GT 红色轮廓
    outline_result = ori_img.copy()

    for polygon in polygons:
        cv2.polylines(
            outline_result,
            [polygon],
            isClosed=True,
            color=GREEN,
            thickness=LINE_WIDTH,
            lineType=cv2.LINE_AA,
        )

    gt_mask = (gt_img > 0).astype(np.uint8)
    gt_contours, _ = cv2.findContours(
        gt_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        outline_result,
        gt_contours,
        contourIdx=-1,
        color=RED,
        thickness=LINE_WIDTH,
        lineType=cv2.LINE_AA,
    )

    stem = Path(filename).stem

    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_2.png"),
        ai_box_mask,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_3.png"),
        filled_result,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_4.png"),
        outline_result,
    )

    return ai_box_mask, outline_result


def vis(image_dir, label_dir, output_dir):
    ai_dir = os.path.join(label_dir, "ai")
    gt_dir = os.path.join(label_dir, "gt")

    os.makedirs(output_dir, exist_ok=True)

    for file in sorted(os.listdir(ai_dir)):
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        label_stem = Path(file).stem
        image_name = f"{label_stem}_0000.png"

        ori_img_path = os.path.join(image_dir, image_name)
        seg_img_path = os.path.join(ai_dir, file)
        gt_img_path = os.path.join(gt_dir, file)

        ori_img = cv2.imread(ori_img_path)
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)

        if ori_img is None:
            print("原图读取失败:", ori_img_path)
            continue
        if seg_img is None:
            print("AI Mask 读取失败:", seg_img_path)
            continue
        if gt_img is None:
            print("GT Mask 读取失败:", gt_img_path)
            continue

        if seg_img.shape != ori_img.shape[:2]:
            seg_img = cv2.resize(
                seg_img,
                (ori_img.shape[1], ori_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if gt_img.shape != ori_img.shape[:2]:
            gt_img = cv2.resize(
                gt_img,
                (ori_img.shape[1], ori_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        output_stem = Path(image_name).stem

        # _0：原图
        cv2.imwrite(
            os.path.join(output_dir, f"{output_stem}_0.png"),
            ori_img,
        )

        # _1：GT 黑白图
        gt_binary = ((gt_img > 0).astype(np.uint8) * 255)
        cv2.imwrite(
            os.path.join(output_dir, f"{output_stem}_1.png"),
            gt_binary,
        )

        frame_mask(
            seg_img,
            gt_img,
            ori_img,
            output_dir,
            image_name,
        )

        print("完成:", file)


vis(
    r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\checking_data\nnUnet2d\imagesTs",
    r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\checking_data\nnUnet2d\labelsTs",
    r"Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\checking_data\nnUnet2d\vis",
)