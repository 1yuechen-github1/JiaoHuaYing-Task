import os
import re
import shutil


def move_data(path):
    for file in os.listdir(path):
        m = re.match(r'^(\d+)', file)
        if not m:
            continue
        num = m.group(1)
        print(num, file)
        os.makedirs(os.path.join(path,str(num)), exist_ok=True)
        shutil.move(os.path.join(path, file),os.path.join(path,str(num), file))


move_data(r'Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\MaxillaryInformation\MaxillaryInformation')


# def move_data(path):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".txt"):
#                 shutil.move(os.path.join(root, file),os.path.join(r'Z:\1.CY-SPACE\JiaoHuaYing\test-data\pcd-txt',file))


# move_data(r'Z:\1.CY-SPACE\JiaoHuaYing\test-data\pcd-txt')