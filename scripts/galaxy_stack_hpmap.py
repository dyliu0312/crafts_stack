"""
This is a script for stacking signals of galaxies in a map using HEALPix pixellation.
"""

import os
from collections import defaultdict
from multiprocessing import Pool

import h5py as h5
import numpy as np
from crafts_stack.hpmap import find_pixels_within_radius, shift_pixel_to_target


# --- HEALPix Utilities ---
def find_frequency_index(freq, freq_bins):
    """
    从频率bin中找到给定频率最接近的索引。

    此函数通过计算目标频率与所有频率bin之间的绝对差值，
    然后返回差值最小的索引，从而确保总是能找到一个最接近的频率。

    参数:
    - freq (float): 要查找的目标频率。
    - freq_bins (numpy.ndarray): 可用的频率bin数组。

    返回:
    - int: `freq_bins` 中最接近频率的索引。
    """
    # 计算目标频率与所有频率bin的绝对差值
    idx = np.abs(freq_bins - freq).argmin()
    return idx

# --- Map Stacking Functions ---
def get_stack_value(results, sort=False):
    """
    从并行堆叠结果中聚合和平均像素值。

    给定一个字典列表，其中每个字典包含一个星系贡献的像素索引和值，
    此函数组合所有结果。对于重叠的像素（即在多个星系贡献中出现），它会计算其平均值。

    参数:
    - results (list): 一个字典列表，每个字典包含键 'pix' (像素索引) 和 'val' (对应值)。

    返回:
    - union_pix (numpy.ndarray): 找到的所有唯一像素索引的排序数组。
    - avg_values (numpy.ndarray): 对应于 `union_pix` 中每个像素的聚合（或平均）值数组。
    """
    pix_values = defaultdict(list)
    for result in results:
        for pix, val in zip(result['pix'], result['val']):
            pix_values[pix].append(val)

    # Calculate average for overlapping pixels and collect results
    union_pix = []
    avg_values = []
    for pix, vals in pix_values.items():
        union_pix.append(pix)
        avg_values.append(np.ma.mean(vals))
    
    if sort:
       # Sort pixels and values to maintain a consistent order
        union_pix = np.array(sorted(union_pix))
        
        # Reorder avg_values to match the sorted union_pix
        pix_order = np.array(list(pix_values.keys()))
        pix_vals = np.array([np.mean(v) for v in pix_values.values()])
        
        # Create a mapping from original pix to sorted index
        map_dict = {p: i for i, p in enumerate(pix_order)}
        sorted_indices = [map_dict[p] for p in union_pix]
        
        avg_values = pix_vals[sorted_indices]

    return np.array(union_pix), np.ma.array(avg_values)


def stack_healpix_map(
    catalog,
    map_pix,
    map_value,
    freq_bins,
    radius_deg=5,
    nside=32,
    nfreq=None,
    lon0=90,
    lat0=0,
):
    """
    处理星系列表以生成堆叠的 HEALPix 地图结果。

    此函数遍历星系目录，找到 HEALPix 地图上对应的信号，将像素索引相对于
    一个共同中心进行归一化，并返回结果以供聚合。

    参数:
    - catalog (list): 包含 'ra'、'dec' 和 'freq' 键的星系字典列表。
    - map_pix (numpy.ndarray): 稀疏天空地图的像素索引数组。
    - map_value (numpy.ndarray): 信号值的二维数组，形状为 `(nfreq, len(map_pix))`。
    - freq_bins (numpy.ndarray): 天空地图的频率bin列表。
    - radius_rad (float, optional): 每个星系周围的角半径，单位为弧度。
                                    默认为5度。
    - nside (int, optional): HEALPix nside 参数。默认为32。
    - nfreq (int, optional): 在目标频率周围平均的相邻频率数量。如果为None，则不进行平均。
    - lon0 (float, optional): 归一化的参考经度。默认为90。
    - lat0 (float, optional): 归一化的参考纬度。默认为0。

    返回:
    - list: 字典列表，每个字典包含归一化后的像素索引 ('pix') 和
            对应于一个星系的信号值 ('val')。
    """
    stack_res = []

    for galaxy in catalog:
        ra, dec, freq = galaxy["ra"], galaxy["dec"], galaxy["freq"]
        ra = np.mod(ra, 360)  # 确保 RA 在 [0, 360) 范围内
        found_pix_ind = find_pixels_within_radius(nside, ra, dec, radius_deg)
        freq_ind = find_frequency_index(freq, freq_bins)

        # get the values of valid pix
        cut_ind = np.isin(map_pix, found_pix_ind)
        found_pix_ind = map_pix[cut_ind]
        
        if nfreq is not None:
            freq_slice = slice(max(freq_ind - nfreq, 0), min(freq_ind + nfreq + 1, len(freq_bins)-1))
            values = map_value[freq_slice, :][:, cut_ind].mean(axis=0)
        else:
            values = map_value[freq_ind, :][cut_ind]

        shifted_pixel = shift_pixel_to_target(
            nside,
            found_pix_ind,
            original_field=(ra, dec),
            new_field=(lon0, lat0)
        )
        
        stack_res.append(
            {
                'pix': shifted_pixel,
                'val': values
            }
        )
    return stack_res

# --- Data Loading and Saving Functions ---
def load_map(
    map_path,
    key_pix="map_pix",
    key_value="clean_map",
    key_freq="freq",
    key_nside="nside",
    swap_axis=False
):
    """
    从 HDF5 文件加载稀疏 HEALPix 地图和相关数据。

    参数:
    - map_path (str): 包含地图数据的 HDF5 文件路径。
    - key_pix (str, optional): 像素索引数据集的键。默认为"map_pix"。
    - key_value (str, optional): 信号值数据集的键。默认为"clean_map"。
    - key_nside (str, optional): HEALPix nside 参数的键。默认为"nside"。
    - key_freq (str, optional): 频率bin数据集的键。默认为"freq"。
    - swap_axis (bool): 如果为True，则交换频率和像素轴。默认为False。

    返回:
    - tuple: 包含以下内容的元组：
        - nside (int): HEALPix nside 参数。
        - map_pix (numpy.ndarray): 像素索引数组。
        - map_value (numpy.ndarray): 信号值的二维数组 (freq, pix)。
        - freq_bins (numpy.ndarray): 频率bin数组。

    """
    with h5.File(map_path, "r") as f:
        nside = f[key_nside][()] # type: ignore
        map_pix = f[key_pix][()] # type: ignore
        freq_bins = f[key_freq][()] # type: ignore
        map_value = f[key_value][()] # type: ignore
    if str(swap_axis).lower() in ("true", "1", "yes"):
        print("Swapping frequency and pixel axes.")
        map_value = np.swapaxes(map_value, 0, 1) # type: ignore
    print(f"Loaded map with nside={nside}, {len(map_pix)} pixels, and {len(freq_bins)} frequency bins.") # type: ignore
    return nside, map_pix, map_value, freq_bins


def load_catalog(cat_path, keys=["ra", "dec", "freq"], cut=None):
    """
    从 HDF5 文件加载星系目录。

    参数:
    - cat_path (str): 包含目录的 HDF5 文件路径。
    - keys (list, optional): 要为每个星系提取的数据集键列表。
                             默认为 ["ra", "dec", "freq"]。

    返回:
    - list: 字典列表，每个字典代表一个星系，并包含指定的键及其值。
    """
    with h5.File(cat_path, "r") as f:
        if cut is not None:
            # 如果提供了切片，则仅加载指定范围内的星系
            catalog = [{key: f[key][i] for key in keys} for i in range(len(f[keys[0]]))[:cut]] # type: ignore
        else:
            catalog = [{key: f[key][i] for key in keys} for i in range(len(f[keys[0]]))] # type: ignore
    print(f"Loaded catalog with {len(catalog)} galaxies.")
    return catalog


def save_result(pix, signal, path, nside):
    """
    将堆叠的 HEALPix 像素索引和值保存到 HDF5 文件。

    参数:
    - pix (numpy.ndarray): 堆叠结果的像素索引。
    - signal (numpy.ndarray): 堆叠结果的信号值。
    - path (str): 输出文件路径。
    - nside (int): HEALPix nside 参数。
    """
    with h5.File(path, "w") as f:
        f.create_dataset("stack_pix", data=pix, dtype="i4")
        f.create_dataset("stack_signal", data=signal, dtype="f2")
        f.create_dataset("nside", data=nside, dtype="i4")


# --- Main Function ---
def main(
    cat_path,
    map_path,
    map_keys={},
    cat_keys={},
    nworker:int = 4,
    radius_deg:float = 1,
    nfreq=None,
    lon0:float = 90,
    lat0:float = 0,
    ouput_path=None,
):
    """
    主函数，用于加载数据、堆叠星系信号并保存结果。

    此函数负责整个流程：加载星系目录和 HEALPix 地图，并行化堆叠过程，
    组合结果，然后将堆叠后的像素和值保存到 HDF5 文件中。

    参数:
    - cat_path (str): 包含星系目录的 HDF5 文件路径。
    - map_path (str): 包含稀疏 HEALPix 地图的 HDF5 文件路径。
    - map_keys (dict, optional): 用于加载地图数据的键字典。
    - cat_keys (dict, optional): 用于加载目录数据的键字典。
    - nworker (int, optional): 用于并行化的工作进程数。 默认为4。
    - radius_deg (float, optional): 用于堆叠的角半径，单位为度。默认为1。
    - nfreq (int, optional): 要平均的相邻频率数量。默认为None。
    - lon0 (float, optional): 堆叠的参考经度。默认为90。
    - lat0 (float, optional): 堆叠的参考纬度。默认为0。
    - ouput_path (str, optional): 保存输出 HDF5 文件的路径。
                                  如果为None，则会自动生成一个默认路径。
    """
    catalog = load_catalog(cat_path, keys=cat_keys.get('keys', ["ra", "dec", "freq"]), cut=cat_keys.get('cut', None))
    nside, map_pix, map_value, freq_bins = load_map(map_path, **map_keys)

    # 将星系目录分割成块以进行并行处理
    chunk_size = len(catalog) // nworker
    galaxy_chunks = [
        catalog[i : i + chunk_size] for i in range(0, len(catalog), chunk_size)
    ]
    # 确保所有星系都包含在块中
    if len(catalog) % nworker != 0:
        galaxy_chunks[-1].extend(catalog[-(len(catalog) % nworker):])

    # 并行处理星系堆叠
    with Pool(nworker) as pool:
        results = pool.starmap(
            stack_healpix_map,
            [
                (
                    chunk,
                    map_pix,
                    map_value,
                    freq_bins,
                    radius_deg,
                    nside,
                    nfreq,
                    lon0,
                    lat0,
                )
                for chunk in galaxy_chunks
            ],
        )
    
    # 组合并行进程的结果
    all_results = [item for sublist in results for item in sublist]
    stack_map_pix, stack_map_value = get_stack_value(all_results)
    
    if ouput_path is None:
        ouput_path = os.path.join(os.path.dirname(map_path), "auto_stack_result.h5")

    save_result(stack_map_pix, stack_map_value, ouput_path, nside)
    print(f"Stacked results saved to: {ouput_path}")

# --- Entry Point ---
if __name__ == "__main__":
    # 从环境变量中读取参数
    cat_path = os.getenv('CAT_PATH')
    map_path = os.getenv('MAP_PATH')

    map_key_values_str = os.getenv("MAP_KEYS")
    map_key_names = ["key_pix", "key_value", "key_freq", "key_nside", "swap_axis"]
    if map_key_values_str is not None:
        try:
            # 拆分并去除空格
            map_key_values_list = [item.strip() for item in map_key_values_str.split(",")]
            # 转换为字典
            map_key_dict = {k: v for k, v in zip(map_key_names, map_key_values_list)}
        except Exception as e:
            raise ValueError(f"Error parsing MAP_KEYS: {map_key_values_str} to {map_key_names}") from e
    else:
        map_key_dict = dict(
            key_pix="map_pix",
            key_value="clean_map",
            key_freq="freq",
            key_nside="nside",
            swap_axis=False,
        )

    cut = os.getenv('CUT')
    if cut is not None:
        cat_keys={'keys':["ra", "dec", "freq"],'cut': int(cut)}
    else:
        cat_keys={'keys':["ra", "dec", "freq"]}

    radius_deg = float(os.getenv('R_DEG', 1.0))
    nfreq = int(os.getenv('NFREQ', -1)) # -1 表示不进行频率平均
    lon0 = float(os.getenv('LON0', 90.0))
    lat0 = float(os.getenv('LAT0', 0.0))
    out_path = os.getenv('OUT_PATH')
    nworker = int(os.getenv('NWORKER', 1))
    print(f"Using {nworker} workers for parallel processing.")

    # 对必需的路径进行基本验证
    if not cat_path or not map_path:
        raise ValueError("请设置 CAT_PATH 和 MAP_PATH 环境变量。")
    
    # 如果未提供或设置为非正值，则将 nfreq 设置为 None
    nfreq = nfreq if nfreq > 0 else None

    # 调用主函数
    main(
        cat_path,
        map_path,
        map_keys=map_key_dict,
        cat_keys=cat_keys,
        nworker=nworker,
        radius_deg=radius_deg,
        nfreq=nfreq,
        lon0=lon0,
        lat0=lat0,
        ouput_path=out_path
    )