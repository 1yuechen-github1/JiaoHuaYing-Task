import os
import numpy as np

def re_scale(path):
    path2 = r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\4.缺牙区(有角化龈)\wash'
    for file in os.listdir(path):
        data = np.loadtxt(os.path.join(path, file))
        coord = data[:, :3]
        colors = data[:, 3:6]
        scale = file.split('.')[0][-1]
        scalars = data[:, 6:7]
        scalars[scalars > 0] = scale
        data_new = np.hstack((coord, colors, scalars))
        np.savetxt(os.path.join(path2, file), data_new)



re_scale(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\4.缺牙区(有角化龈)\txt')