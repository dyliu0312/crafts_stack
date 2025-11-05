from typing import List, Tuple

import numpy as np


def calculate_signal_background_stats(
    data: List[np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算信号和背景区域的统计量

    参数:
        data: 输入数据列表
        mask: 布尔掩码

    返回:
        signal_means: 信号均值列表
        signal_stds: 信号标准差列表
        background_means: 背景均值列表
        background_stds: 背景标准差列表
    """

    signal_means = []
    signal_stds = []
    background_means = []
    background_stds = []

    for arr in data:
        if arr.shape != mask.shape:
            raise ValueError(f"数据形状 {arr.shape} 与mask形状 {mask.shape} 不匹配")

        # 计算信号统计量
        signal_data = arr[mask]
        signal_means.append(np.mean(signal_data))
        signal_stds.append(np.std(signal_data))

        # 计算背景统计量
        background_data = arr[~mask]
        background_means.append(np.mean(background_data))
        background_stds.append(np.std(background_data))

    return (
        np.array(signal_means),
        np.array(signal_stds),
        np.array(background_means),
        np.array(background_stds),
    )
