import os
import numpy as np
import open3d as o3d
import re
import matplotlib.pyplot as plt


path = r"C:\yuechen\code\jiaohuaying\1.code\0105\data\DATA\label"
for file in os.listdir(path):
    label = file.split('_')[-1]
    label = label.split('.')[0] 
    dianyun = np.loadtxt(os.path.join(path,file))
    n_points = dianyun.shape[0] 
    label_column = np.full((n_points, 1), label) 
    dianyun_with_label = np.hstack([dianyun, label_column])
    # 在末尾添加标签列
    dianyun_with_label = np.hstack([dianyun, label_column])
