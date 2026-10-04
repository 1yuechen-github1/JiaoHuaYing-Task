
import os
import shutil
fold = r'E:\CY\HumanVsMachineMatch\角化龈分割结果_csj\分割结果-CSJ'
for file in os.listdir(fold):
    file_dir = os.path.join(fold,file)
    for file2 in os.listdir(file_dir):
        if file2.endswith('.ply'):
            shutil.move(os.path.join(file_dir,file2),os.path.join(fold,file2))