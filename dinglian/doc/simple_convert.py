import numpy as np
from pathlib import Path
import os

def convert_each_top_level_directory(source_dir, output_dir=None):
    """
    将每个顶级目录下的所有txt文件合并为一个npy文件集合
    例如：0001/下的所有txt -> 0001/color.npy, coord.npy等
    """
    source_path = Path(source_dir)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = source_path
    else:
        output_dir = Path(output_dir)
    
    # 获取所有顶级目录
    top_level_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not top_level_dirs:
        print(f"在目录 {source_dir} 中没有找到数字命名的顶级目录")
        return
    
    print(f"找到 {len(top_level_dirs)} 个顶级目录")
    print("数据格式: x,y,z, r,g,b, scalar")
    print("Segment标签规则: scalar > 0 -> 1, scalar == 0 -> 0")
    
    # 处理每个顶级目录
    for dir_idx, top_dir in enumerate(top_level_dirs):
        print(f"\n处理目录 {dir_idx + 1}/{len(top_level_dirs)}: {top_dir.name}")
        
        # 递归获取该目录下的所有txt文件
        txt_files = list(top_dir.rglob("*.txt"))
        
        if not txt_files:
            print(f"  在目录 {top_dir.name} 中没有找到txt文件")
            continue
        
        print(f"  找到 {len(txt_files)} 个txt文件")
        
        # 存储所有数据
        all_coord = []
        all_color = []
        all_scalar = []
        
        # 读取所有txt文件
        for file_idx, txt_file in enumerate(txt_files):
            try:
                # 读取txt文件
                point_data = np.loadtxt(txt_file)
                
                # 检查数据维度
                if point_data.ndim != 2:
                    print(f"  警告: 文件 {txt_file.name} 数据维度不正确，跳过")
                    continue
                
                n_points, n_features = point_data.shape
                
                if n_features != 7:
                    print(f"  警告: 文件 {txt_file.name} 有 {n_features} 个特征，期望7个")
                    if n_features < 7:
                        print(f"    错误: 特征数不足7个，跳过该文件")
                        continue
                    else:
                        point_data = point_data[:, :7]  # 只取前7列
                
                # 分割数据
                coord = point_data[:, 0:3]    # x, y, z
                color = point_data[:, 3:6]    # r, g, b
                scalar = point_data[:, 6]     # scalar值
                
                all_coord.append(coord)
                all_color.append(color)
                all_scalar.append(scalar)
                
                print(f"    读取: {txt_file.relative_to(top_dir)} - {n_points} 个点")
                
            except Exception as e:
                print(f"    处理文件 {txt_file.name} 时出错: {e}")
                continue
        
        if not all_coord:
            print(f"  没有成功读取任何文件，跳过目录 {top_dir.name}")
            continue
        
        # 合并所有数据
        try:
            merged_coord = np.vstack(all_coord)
            merged_color = np.vstack(all_color)
            merged_scalar = np.concatenate(all_scalar)
            
            n_total_points = len(merged_coord)
            print(f"  合并完成: 总共 {n_total_points} 个点")
            
            # 根据scalar值计算segment标签
            segment = np.where(merged_scalar > 0, 1, 0).astype(np.int32)
            
            # 统计segment分布
            unique_segments, segment_counts = np.unique(segment, return_counts=True)
            segment_info = ", ".join([f"类别{seg}: {count}点" for seg, count in zip(unique_segments, segment_counts)])
            print(f"  Segment分布: {segment_info}")
            
            # 创建其他必要的数组
            normal = np.zeros((n_total_points, 3))      # 法向量（默认零向量）
            instance = np.zeros(n_total_points)         # 实例标签（默认0）
            
            # 在顶级目录中创建npy文件
            np.save(top_dir / "color.npy", merged_color.astype(np.float32))
            np.save(top_dir / "coord.npy", merged_coord.astype(np.float32))
            np.save(top_dir / "normal.npy", normal.astype(np.float32))
            np.save(top_dir / "instance.npy", instance.astype(np.int32))
            np.save(top_dir / "segment.npy", segment.astype(np.int32))
            # np.save(top_dir / "scalar.npy", merged_scalar.astype(np.float32))
            
            print(f"  保存到: {top_dir.name}/")
            print(f"  坐标范围: x[{merged_coord[:,0].min():.3f}, {merged_coord[:,0].max():.3f}]")
            print(f"  颜色范围: r[{merged_color[:,0].min():.3f}, {merged_color[:,0].max():.3f}]")
            print(f"  Scalar范围: [{merged_scalar.min():.3f}, {merged_scalar.max():.3f}]")
            
        except Exception as e:
            print(f"  合并数据时出错: {e}")
    
    print(f"\n转换完成！每个目录下都生成了npy文件")

def convert_to_separate_output(source_dir, output_base_dir=None):
    """
    将每个顶级目录转换到单独的输出目录
    """
    source_path = Path(source_dir)
    
    if output_base_dir is None:
        output_base_dir = source_path / "npy_output"
    else:
        output_base_dir = Path(output_base_dir)
    
    output_base_dir.mkdir(exist_ok=True)
    
    # 获取所有顶级目录
    top_level_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for top_dir in top_level_dirs:
        print(f"\n处理目录: {top_dir.name}")
        
        # 递归获取该目录下的所有txt文件
        txt_files = list(top_dir.rglob("*.txt"))
        
        if not txt_files:
            print(f"  没有找到txt文件，跳过")
            continue
        
        # 创建输出目录
        output_dir = output_base_dir / top_dir.name
        output_dir.mkdir(exist_ok=True)
        
        # 存储所有数据
        all_coord = []
        all_color = []
        all_scalar = []
        
        for txt_file in txt_files:
            try:
                data = np.loadtxt(txt_file)
                if data.ndim == 2 and data.shape[1] >= 7:
                    if data.shape[1] > 7:
                        data = data[:, :7]
                    all_coord.append(data[:, 0:3])
                    all_color.append(data[:, 3:6])
                    all_scalar.append(data[:, 6])
                    print(f"  读取: {txt_file.relative_to(top_dir)}")
            except Exception as e:
                print(f"  读取文件出错: {e}")
        
        if not all_coord:
            print(f"  没有成功读取任何文件")
            continue
        # for txt_file in txt_files:
        #     try:
        #         data = np.loadtxt(txt_file)
        #         if data.ndim == 2 and data.shape[1] >=5:
        #             if data.shape[1] > 6:
        #                 data = data[:, :7]
        #             all_coord.append(data[:, 0:3])
        #             all_color.append(data[:, 3:6])
        #             # all_scalar.append(data[:, 6])
        #             print(f"  读取: {txt_file.relative_to(top_dir)}")
        #     except Exception as e:
        #         print(f"  读取文件出错: {e}")
        
        # if not all_coord:
        #     print(f"  没有成功读取任何文件")
        #     continue
        
        # 合并数据
        merged_coord = np.vstack(all_coord)
        merged_color = np.vstack(all_color)
        merged_scalar = np.concatenate(all_scalar)
        points = len(all_coord[0])
        label_column = np.full((points, 1), 0) 
        # 计算segment
        segment = np.where(merged_scalar > 0, 1, 0).astype(np.int32)
        
        # 创建其他数组
        normal = np.zeros((len(merged_coord), 3))
        instance = np.zeros(len(merged_coord))
        
        # 保存文件
        np.save(output_dir / "color.npy", merged_color.astype(np.float32))
        np.save(output_dir / "coord.npy", merged_coord.astype(np.float32))
        np.save(output_dir / "normal.npy", normal.astype(np.float32))
        np.save(output_dir / "instance.npy", instance.astype(np.int32))
        np.save(output_dir / "segment.npy", segment.astype(np.int32))
        # np.save(output_dir / "label.npy", merged_scalar.astype(np.float32))
        np.save(output_dir / "label.npy", label_column.astype(np.float32))
        
        print(f"  保存到: {output_dir.relative_to(output_base_dir)}")
        print(f"  总点数: {len(merged_coord)}")

def analyze_top_level_directories(source_dir):
    """
    分析每个顶级目录的数据
    """
    source_path = Path(source_dir)
    
    top_level_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print("目录分析:")
    print("=" * 50)
    
    for top_dir in top_level_dirs:
        txt_files = list(top_dir.rglob("*.txt"))
        total_points = 0
        file_count = 0
        
        print(f"\n{top_dir.name}/:")
        print(f"  TXT文件数: {len(txt_files)}")
        
        # 抽样分析
        for txt_file in txt_files[:3]:  # 只分析前3个文件
            try:
                data = np.loadtxt(txt_file)
                if data.ndim == 2 and data.shape[1] >= 7:
                    points = data.shape[0]
                    total_points += points
                    file_count += 1
                    
                    scalars = data[:, 6]
                    positive_ratio = np.sum(scalars > 0) / len(scalars) * 100
                    
                    print(f"    {txt_file.name}: {points} 点, 正值: {positive_ratio:.1f}%")
            except:
                pass
        
        if file_count > 0:
            avg_points = total_points / file_count
            estimated_total = avg_points * len(txt_files)
            print(f"  估计总点数: ~{estimated_total:.0f}")

if __name__ == "__main__":
    # 设置源目录路径
    source_directory = r"C:\yuechen\code\jiaohuaying\1.code\0105\data\DATA\label\test"
    
    print("每个顶级目录下的所有TXT文件将合并为一组NPY文件")
    print("Segment标签规则: scalar > 0 -> 1, scalar == 0 -> 0")
    
    # 检查目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 目录 {source_directory} 不存在")
        exit(1)
    
    print("请选择操作:")
    print("1. 分析目录结构")
    print("2. 在原目录中生成NPY文件")
    print("3. 在新目录中生成NPY文件")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        analyze_top_level_directories(source_directory)
    elif choice == "2":
        convert_each_top_level_directory(source_directory)
    elif choice == "3":
        convert_to_separate_output(source_directory)
    else:
        print("无效选择")