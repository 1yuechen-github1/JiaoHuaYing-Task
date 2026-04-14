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


# def move_data(path):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".txt"):
#                 shutil.move(os.path.join(root, file),os.path.join(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\4.缺牙区(有角化龈)\txt',file))
#

move_data(r'C:\yuechen\code\jiaohuaying\2.data\3.0326_data\wash\5.缺牙区-有角化龈\2.缺牙区')