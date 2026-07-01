
import os
import json
import numpy as np
import cv2
from skimage import measure
from PIL import Image
import shutil


def count_class_pixels(label_dir, class_ids):
    """统计每个类别的像素数量"""
    counts = {str(cid): 0 for cid in class_ids}
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        for cid in class_ids:
            counts[str(cid)] += np.sum(label_img == cid)
    return counts

def split_cases_by_class(label_dir, class_ids, train_ratio=0.7, val_ratio=0.2):
    """按类别分别划分训练、验证和测试索引，比例为7:2:1"""
    
    # 获取所有标签文件
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    
    # 为每个类别创建索引列表
    class_indices = {cid: [] for cid in class_ids}
    
    # 遍历所有标签文件，统计每个文件包含哪些类别
    for idx, label_file in enumerate(label_files):
        label_path = os.path.join(label_dir, label_file)
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        
        for cid in class_ids:
            if np.any(label_img == cid):  # 如果这个文件包含该类别
                class_indices[cid].append(idx)
    
    # 为每个类别单独划分，确保每个类别在三个集合中都有样本
    train_indices = set()
    val_indices = set()
    test_indices = set()
    
    for cid, indices in class_indices.items():
        if len(indices) == 0:
            print(f"警告: 类别 {cid} 没有在任何图像中出现")
            continue
        
        np.random.shuffle(indices)
        total = len(indices)
        
        # 计算分割点
        train_split = max(1, int(total * train_ratio))  # 确保至少一个训练样本
        val_split = train_split + max(1, int(total * val_ratio))  # 确保至少一个验证样本
        
        # 确保测试集至少有一个样本
        if val_split >= total:
            val_split = total - 1
            train_split = val_split - 1 if val_split > 1 else 1
        
        # 划分三个集合
        train_indices.update(indices[:train_split])
        val_indices.update(indices[train_split:val_split])
        test_indices.update(indices[val_split:])
    
    # 转换为列表并排序
    train_list = sorted(list(train_indices))
    val_list = sorted(list(val_indices))
    test_list = sorted(list(test_indices))
    
    print(f"\n类别划分统计:")
    for cid in class_ids:
        train_count = len([idx for idx in train_list if idx in class_indices[cid]])
        val_count = len([idx for idx in val_list if idx in class_indices[cid]])
        test_count = len([idx for idx in test_list if idx in class_indices[cid]])
        total_count = len(class_indices[cid])
        print(f"类别 {cid}: 训练集 {train_count}/{total_count} ({train_count/total_count:.1%}), "
              f"验证集 {val_count}/{total_count} ({val_count/total_count:.1%}), "
              f"测试集 {test_count}/{total_count} ({test_count/total_count:.1%})")
    
    # 验证每个类别在三个集合中都有样本
    for cid in class_ids:
        train_has_class = any(idx in train_list for idx in class_indices[cid])
        val_has_class = any(idx in val_list for idx in class_indices[cid])
        test_has_class = any(idx in test_list for idx in class_indices[cid])
        
        if not train_has_class:
            print(f"警告: 类别 {cid} 在训练集中没有样本，将添加一个样本")
            idx_to_add = np.random.choice(class_indices[cid])
            train_list.append(idx_to_add)
            
        if not val_has_class:
            print(f"警告: 类别 {cid} 在验证集中没有样本，将添加一个样本")
            idx_to_add = np.random.choice(class_indices[cid])
            val_list.append(idx_to_add)
            
        if not test_has_class:
            print(f"警告: 类别 {cid} 在测试集中没有样本，将添加一个样本")
            idx_to_add = np.random.choice(class_indices[cid])
            test_list.append(idx_to_add)
    
    # 去重并排序
    train_list = sorted(list(set(train_list)))
    val_list = sorted(list(set(val_list)))
    test_list = sorted(list(set(test_list)))
    
    # 确保三个集合之间没有重叠
    train_set = set(train_list)
    val_set = set(val_list)
    test_set = set(test_list)
    
    # 处理训练集和验证集重叠
    train_val_overlap = train_set & val_set
    if train_val_overlap:
        print(f"警告: 训练集和验证集有 {len(train_val_overlap)} 个重叠样本，将自动调整")
        val_set = val_set - train_val_overlap
    
    # 处理训练集和测试集重叠
    train_test_overlap = train_set & test_set
    if train_test_overlap:
        print(f"警告: 训练集和测试集有 {len(train_test_overlap)} 个重叠样本，将自动调整")
        test_set = test_set - train_test_overlap
    
    # 处理验证集和测试集重叠
    val_test_overlap = val_set & test_set
    if val_test_overlap:
        print(f"警告: 验证集和测试集有 {len(val_test_overlap)} 个重叠样本，将自动调整")
        test_set = test_set - val_test_overlap
    
    # 更新列表
    train_list = sorted(list(train_set))
    val_list = sorted(list(val_set))
    test_list = sorted(list(test_set))
    
    print(f"\n最终划分结果:")
    print(f"训练集数量: {len(train_list)}")
    print(f"验证集数量: {len(val_list)}")
    print(f"测试集数量: {len(test_list)}")
    
    return train_list, val_list, test_list



def create_splits_json(output_dir, case_names, train_idx, val_idx,test_ids):
    """生成 splits_final.json"""
    splits = [{
        "train": [case_names[i] for i in train_idx],
        "val": [case_names[i] for i in val_idx],
        "test": [case_names[i] for i in test_ids]
    }]
    splits_path = os.path.join(output_dir, "splits_final.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)
    print(f"已生成 splits_final.json，训练集: {len(train_idx)}，验证集: {len(val_idx)}")


def labelme_to_nnunet(labelme_dir, output_dir, dataset_name="MyDataset"):
    """
    将LabelMe标注格式转换为nnU-Net格式
    
    参数:
    labelme_dir: 包含原始图像和labelme json文件的文件夹
    output_dir: 输出nnU-Net数据集的根目录
    dataset_name: 数据集名称
    """
    
    # 创建输出目录结构
    images_tr_dir = os.path.join(output_dir, dataset_name, 'imagesTr')
    labels_tr_dir = os.path.join(output_dir, dataset_name, 'labelsTr')
    images_ts_dir = os.path.join(output_dir, dataset_name, 'imagesTs')
    
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    os.makedirs(images_ts_dir, exist_ok=True)
    
    # 获取所有json文件
    json_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 用于存储所有类别名称
    all_class_names = set()
    case_names = []  # 存储所有case的名称（原始文件名）
    processed_count = 0
    
    for i, json_file in enumerate(json_files):
        json_path = os.path.join(labelme_dir, json_file)
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取原始文件名（不含扩展名）
        base_name = os.path.splitext(json_file)[0]
        case_names.append(base_name)  # 保存原始文件名
        image_file = data['imagePath']
        
        # 修复图像路径问题 - 处理相对路径 ..
        image_path = None
        
        # 方法1: 处理 ..\ 相对路径
        if image_file.startswith('..\\'):
            # 获取label文件夹的父目录
            parent_dir = os.path.dirname(labelme_dir)
            # 构建完整路径
            relative_path = image_file[3:]  # 去掉 ..\
            image_path = os.path.join(parent_dir, relative_path)
        
        # 方法2: 在label文件夹中查找同名文件
        if image_path is None or not os.path.exists(image_path):
            possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG']
            for ext in possible_extensions:
                # 使用JSON文件名 + 扩展名
                alt_path = os.path.join(labelme_dir, base_name + ext)
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        # 方法3: 使用imagePath中的文件名（去掉路径）
        if image_path is None or not os.path.exists(image_path):
            filename_only = os.path.basename(image_file)
            alt_path = os.path.join(labelme_dir, filename_only)
            if os.path.exists(alt_path):
                image_path = alt_path
        
        if image_path is None or not os.path.exists(image_path):
            print(f"警告: 找不到图像文件 {image_file}，跳过 {json_file}")
            print(f"尝试的路径: {image_path}")
            continue
        
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"警告: 无法读取图像 {image_path}，跳过")
            continue
        
        height, width = original_image.shape[:2]
        
        # 创建空的标签图像（全0，表示背景）
        label_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 处理每个形状/标注
        for shape in data['shapes']:
            label_name = shape['label']
            
            # 记录类别名称（即使是数字也记录）
            all_class_names.add(label_name)
            
            # 将标签名称转换为整数ID
            try:
                class_id = int(label_name)
            except ValueError:
                # 如果标签不是数字，使用映射关系
                if label_name not in class_name_to_id:
                    class_name_to_id[label_name] = len(class_name_to_id) + 1
                class_id = class_name_to_id[label_name]
            
            points = shape['points']
            # 将点转换为多边形掩码
            polygon_points = np.array(points, dtype=np.int32)
            cv2.fillPoly(label_mask, [polygon_points], class_id)
        
        # 生成nnU-Net格式的文件名
        nnunet_image_name = f"{base_name}_0000.png"  # 图像格式: 原始文件名_0000_0000.png
        nnunet_label_name = f"{base_name}.png"       # 标签格式: 原始文件名_0000.png
        
        # 保存图像到imagesTr（复制原始图像）
        output_image_path = os.path.join(images_tr_dir, nnunet_image_name)
        cv2.imwrite(output_image_path, original_image)
        
        # 保存标签到labelsTr
        output_label_path = os.path.join(labels_tr_dir, nnunet_label_name)
        print("label_mask unique values:", np.unique(label_mask))
        cv2.imwrite(output_label_path, label_mask)
        
        # 可视化标签（可选）
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(vis_dir, nnunet_label_name.replace('.png', '_vis.png')), label_mask, cmap='jet')

        print(f"处理完成: {base_name}")
        processed_count += 1
    
    # 创建dataset.json文件
    create_dataset_json(
        output_dir=os.path.join(output_dir, dataset_name),
        dataset_name=dataset_name,
        class_names=sorted(list(all_class_names)),
        num_training_cases=processed_count,
        case_names=case_names
    )

    # 统计类别像素数量
    class_ids = [1, 2, 3]
    label_dir = os.path.join(output_dir, dataset_name, 'labelsTr')
    class_counts = count_class_pixels(label_dir, class_ids)
    print("每个类别的像素数量：", class_counts)

    num_cases = processed_count
    train_idx, val_idx,test_ids = split_cases_by_class(label_dir, class_ids,0.7,0.2)
    create_splits_json(os.path.join(output_dir, dataset_name), case_names, train_idx, val_idx, test_ids)

    
    print(f"\n转换完成！")
    print(f"成功处理 {processed_count}/{len(json_files)} 个文件")
    print(f"数据集保存在: {os.path.join(output_dir, dataset_name)}")
    print(f"发现 {len(all_class_names)} 个类别: {sorted(list(all_class_names))}")


def create_dataset_json(output_dir, dataset_name, class_names, num_training_cases, case_names):
    """创建nnU-Net需要的dataset.json文件"""
    
    # 构建标签名称映射 - 使用预定义的类别名称
    labels = {"0": "background", "1": "shangqiangya", "2": "shanghouya", "3": "xiaheya"}
    
    # 动态添加发现的类别
    for i, class_name in enumerate(class_names, 1):
        if str(i) not in labels:  # 如果预定义中没有这个编号，则添加
            labels[str(i)] = class_name
    
    dataset_info = {
        "name": dataset_name,
        "description": f"{dataset_name} dataset converted from LabelMe",
        "reference": "Converted by labelme_to_nnunet script",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "numTraining": num_training_cases,
        "modality": {
            "0": "RGB"
        },
        "labels": labels,
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{case_names[i]}_0000.png",
                "label": f"./labelsTr/{case_names[i]}.png"
            } for i in range(num_training_cases)
        ],
        "test": []
    }
    
    json_path = os.path.join(output_dir, 'dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    print(f"已创建 dataset.json")

def verify_dataset(dataset_dir):
    """验证生成的数据集是否正确"""
    print("\n正在验证数据集...")
    
    images_dir = os.path.join(dataset_dir, 'imagesTr')
    labels_dir = os.path.join(dataset_dir, 'labelsTr')
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    print(f"图像文件数: {len(image_files)}")
    print(f"标签文件数: {len(label_files)}")
    
    # 检查文件名对应关系
    for img_file in image_files:
        base_name = img_file.split('_0000.png')[0]  # 从"原始文件名_0000_0000.png"中提取"原始文件名_0000"
        expected_label = f"{base_name}.png"
        
        if expected_label not in label_files:
            print(f"错误: 找不到对应的标签文件 {expected_label}")
            return False
    
    # 检查标签内容
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        unique_values = np.unique(label_img)
        print(f"标签 {label_file} 中的像素值: {unique_values}")
    
    print("验证通过！数据集格式正确。")
    return True

if __name__ == "__main__":
    # 使用示例
    labelme_directory = r"Z:\1.CY-SPACE\JiaoHuaYing\1.AllData-PNG\linshi\nnunet"  # 替换为你的LabelMe数据文件夹路径
    output_directory = r"Z:\1.CY-SPACE\JiaoHuaYing\1.AllData-PNG\linshi\nnunet_output"  # 替换为输出路径
    dataset_name = "kousao"           # 你的数据集名称
    
    # 初始化类别映射字典
    class_name_to_id = {}
    
    # 执行转换
    labelme_to_nnunet(labelme_directory, output_directory, dataset_name)
    
    # 验证数据集
    dataset_path = os.path.join(output_directory, dataset_name)
    verify_dataset(dataset_path)