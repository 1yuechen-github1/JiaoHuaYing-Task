import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def vis(path, path2, path3):
    for file in os.listdir(path2):
        filename = file.split('.')[0]
        filename = filename + '_0000.png'
        ori_img_path = os.path.join(path, filename)
        seg_img_path = os.path.join(path2, file)
        print(file, filename)

        # 读取图像
        ori_img = cv2.imread(ori_img_path)
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        seg_img1 = seg_img
        seg_img = (seg_img > 0).astype(np.uint8)  # 二值化分割图
        mask = seg_img.astype(np.uint8)
        masked_img = cv2.bitwise_and(ori_img, ori_img, mask=mask)
        # cv2.imshow("Masked Image", masked_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(path3,filename), masked_img)


# 调用可视化函数
vis(r'C:\yuechen\code\jiaohuaying\2.data\0408\nnunet\imagesTs',
    r'C:\yuechen\code\jiaohuaying\2.data\0408\nnunet\labelTs\gt',
    r'C:\yuechen\code\jiaohuaying\2.data\0408\nnunet\vis\gt')
