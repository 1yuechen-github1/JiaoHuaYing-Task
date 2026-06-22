import os
import cv2
import numpy as np


def corrosion(path, save_path, kernel_size=3):
    os.makedirs(save_path, exist_ok=True)

    # 腐蚀核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for file in os.listdir(path):
        img_path = os.path.join(path, file)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"跳过无法读取: {file}")
            continue

        # 如果是彩色图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # =========================
        # 1. 提取像素 > 0 的区域 mask
        # =========================
        mask = (gray > 0).astype(np.uint8) * 255

        # =========================
        # 2. 对 mask 做腐蚀
        # =========================
        eroded_mask = cv2.erode(mask, kernel, iterations=19)

        # =========================
        # 3. 贴回原图（只保留腐蚀后的区域）
        # =========================
        if len(img.shape) == 3:
            result = img.copy()
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (eroded_mask > 0)
        else:
            result = gray * (eroded_mask > 0)

        # =========================
        # 4. 保存
        # =========================
        save_path_file = os.path.join(save_path, file)
        cv2.imwrite(save_path_file, result)

        print(f"已处理: {file}")


if __name__ == "__main__":
    path = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\nnunet-2d\INT"
    save_path = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\nnunet-2d\PUT"

    corrosion(path, save_path, kernel_size=3)