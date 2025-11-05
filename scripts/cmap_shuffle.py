"""
A script to shuffle the frequency slices of cuboid map, and also copy the rest datasets.
"""
import os
import sys
import h5py
import numpy as np
from typing import List, Any, Optional

# --- 1. 辅助函数：处理属性和数据复制 ---

def copy_attributes(source_obj: Any, dest_obj: Any):
    """Copies all HDF5 attributes from a source object to a destination object."""
    for key, value in source_obj.attrs.items():
        try:
            dest_obj.attrs[key] = value
        except Exception as e:
            print(f"Warning: Could not copy attribute '{key}'. Error: {e}")

def get_dataset(h5_object: h5py.File, key: str) -> Optional[Any]:
    """Safely retrieves an object (Group or Dataset) from an HDF5 file."""
    try:
        return h5_object[key]
    except KeyError:
        return None

def copy_h5_item_recursive(source_file: h5py.File, dest_file: h5py.File, key: str):
    """
    Recursively copies a Group or Dataset from source_file to dest_file,
    preserving attributes.
    """
    source_obj = get_dataset(source_file, key)
    if source_obj is None:
        return

    if isinstance(source_obj, h5py.Group):
        # Create the group in the destination file
        dest_group = dest_file.create_group(key)
        copy_attributes(source_obj, dest_group)
        
        # Recursively copy members
        for name in source_obj.keys():
            copy_h5_item_recursive(source_file, dest_file, f"{key}/{name}")
            
    elif isinstance(source_obj, h5py.Dataset):
        # Copy the dataset directly
        source_file.copy(source_obj, dest_file, name=key)
        # Note: h5py.File.copy usually preserves attributes, but we ensure it.
        dest_dset = get_dataset(dest_file, key)
        if dest_dset:
            copy_attributes(source_obj, dest_dset)


# --- 2. 主要打乱和复制逻辑 ---

def process_shuffle(
    input_filepath: str,
    output_filepath: str,
    shuffle_keys: List[str],
    copy_keys: List[str],
    shuffle_axis: int = 0
):
    """
    Performs shuffle on specified 3D datasets and copies other items.

    Args:
        input_filepath (str): Path to the source HDF5 file.
        output_filepath (str): Path to the destination HDF5 file.
        shuffle_keys (List[str]): List of HDF5 dataset paths (keys) to be shuffled.
        copy_keys (List[str]): List of HDF5 object paths (keys) to be copied directly.
        shuffle_axis (int): The axis along which to perform the shuffle (default: 0).
    """
    print("--- Starting HDF5 Processing ---")
    print(f"Input: {input_filepath}")
    print(f"Output: {output_filepath}")
    print(f"Shuffle Keys: {shuffle_keys}")
    print(f"Copy Keys: {copy_keys}")
    
    # 集合用于快速查找和检查是否已处理
    all_processed_keys = set()

    try:
        # 打开输入文件和创建/覆盖输出文件
        with h5py.File(input_filepath, 'r') as fin, \
             h5py.File(output_filepath, 'w') as fout:
            
            # 1. 处理文件根目录属性 (Attributes)
            copy_attributes(fin, fout)
            
            # 2. 执行打乱 (Shuffle)
            if shuffle_keys:
                print("\n--- Executing Shuffle ---")
                
                # 获取一个数据集，读取其大小，并生成打乱索引
                first_dset_key = shuffle_keys[0]
                first_dset = get_dataset(fin, first_dset_key)
                
                if first_dset is None or not isinstance(first_dset, h5py.Dataset) or first_dset.ndim < 3:
                    print(f"Error: First shuffle dataset '{first_dset_key}' is not a valid 3D dataset.")
                    return

                num_slices = first_dset.shape[shuffle_axis]
                shuffle_indices = np.arange(num_slices)
                np.random.shuffle(shuffle_indices)
                print(f"Generated {num_slices} shuffle indices on axis {shuffle_axis}.")
                
                # 对所有需要打乱的数据集应用相同的索引
                for dset_key in shuffle_keys:
                    dset_in = get_dataset(fin, dset_key)
                    if dset_in is None or not isinstance(dset_in, h5py.Dataset):
                        print(f"Warning: Dataset '{dset_key}' not found or is not a Dataset. Skipping.")
                        continue
                    
                    if dset_in.ndim < 3:
                        print(f"Warning: Dataset '{dset_key}' is not 3D. Skipping.")
                        continue

                    print(f"Shuffling dataset: {dset_key} (Shape: {dset_in.shape})")
                    
                    # 读取整个数据集到内存 (注意：对于超大数据集可能需要分块读写)
                    data = dset_in[:]
                    
                    # 创建切片对象以应用打乱索引
                    # 例如，如果 shuffle_axis=0, 切片是 [shuffle_indices, :, :]
                    slices = [slice(None)] * dset_in.ndim
                    slices[shuffle_axis] = shuffle_indices
                    
                    shuffled_data = data[tuple(slices)]
                    
                    # 在输出文件中创建并写入打乱后的数据集
                    dset_out = fout.create_dataset(dset_key, data=shuffled_data, compression="gzip")
                    copy_attributes(dset_in, dset_out)
                    all_processed_keys.add(dset_key.strip('/'))


            # 3. 复制指定的数据 (Groups and Datasets)
            if copy_keys:
                print("\n--- Executing Direct Copy ---")
                for key in copy_keys:
                    path = key.strip('/')
                    if path in all_processed_keys:
                        print(f"Warning: Item '{key}' already processed (shuffled). Skipping copy.")
                        continue
                        
                    print(f"Copying item: {key}")
                    
                    # 递归复制 Group 或 Dataset
                    copy_h5_item_recursive(fin, fout, path)
                    all_processed_keys.add(path)
                    
            # 4. 最终检查 (可选：可以遍历文件剩余未处理的对象并复制)
            # 在此简化版本中，我们只处理明确指定的 keys
            
            print(f"\nSuccessfully processed file and saved to {output_filepath}")

    except Exception as e:
        print(f"Fatal Error during HDF5 processing: {e}")
        sys.exit(1)


# --- 3. 脚本入口点 ---

if __name__ == "__main__":
    # 从环境变量中读取配置
    # 注意：在 Linux/macOS 中，列表元素通常用逗号分隔
    INPUT_FILE = os.environ.get('INPUT_PATH')
    OUTPUT_FILE = os.environ.get('OUTPUT_PATH')
    SHUFFLE_KEYS_STR = os.environ.get('SHUFFLE_KEYS', '')
    COPY_KEYS_STR = os.environ.get('COPY_KEYS', '')
    SHUFFLE_AXIS = int(os.environ.get('SHUFFLE_AXIS', '0'))
    
    # 验证输入
    if not INPUT_FILE or not OUTPUT_FILE:
        print("Error: INPUT_PATH and OUTPUT_PATH environment variables must be set.")
        sys.exit(1)

    # 转换 keys 字符串为列表
    shuffle_keys = [k.strip() for k in SHUFFLE_KEYS_STR.split(',') if k.strip()]
    copy_keys = [k.strip() for k in COPY_KEYS_STR.split(',') if k.strip()]

    if not shuffle_keys and not copy_keys:
        print("Warning: Neither SHUFFLE_KEYS nor COPY_KEYS are provided. Nothing will be processed.")
        # exit
        sys.exit(0)
    
    # 执行处理
    # 假设您的 3D 数据是 [时间/切片, 宽度, 高度]，所以默认打乱第一个维度 (0)
    process_shuffle(
        input_filepath=INPUT_FILE,
        output_filepath=OUTPUT_FILE,
        shuffle_keys=shuffle_keys,
        copy_keys=copy_keys,
        shuffle_axis=SHUFFLE_AXIS
    )