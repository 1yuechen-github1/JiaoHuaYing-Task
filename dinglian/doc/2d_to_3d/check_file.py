import os


def check_file(path):
    for file in os.listdir(path):
        if file.endswith(".txt.txt"):
            file.split(".")[0]
            class_name = file.split(".")[0] + '.txt'

            os.rename(os.path.join(path,file),os.path.join(path,class_name))
            # print(os.path.join(path,class_name))

check_file(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\4.缺牙区(有角化龈)\txt')