"""
To plot HEALPix maps easily.
Assuming **RING** order**
"""

from collections.abc import Iterable
from typing import List, Tuple, Union, Optional

import h5py as h5
import healpy as hp
import numpy as np


def gen_random_hpmap(nside: int, seed: int = 42) -> np.ma.MaskedArray:
    """
    Generate a random Healpix map with a given nside and seed.

    Parameters
    ----------
    nside : int
        The Healpix nside resolution parameter.
    seed : int, optional
        The seed for the random number generator (default: 42).

    Returns
    -------
    healpix_map : np.ma.MaskedArray
        A random Healpix map with the given nside and seed.
    """
    np.random.seed(seed)
    npix = hp.nside2npix(nside)
    healpix_map = np.random.rand(npix)
    return hp.ma(healpix_map)

def cut_hpmap(
    nside: int, 
    pixels: Optional[Iterable[int]] = None, 
    values: Union[Iterable[float], np.ndarray, np.ma.MaskedArray, None] = None,
    ra_range: Optional[List[float]] = None, 
    dec_range: Optional[List[float]] = None,
    freq_first: bool = False
):
    if pixels is None:
        pixels = np.arange(hp.nside2npix(nside)) 
    else:
        pixels = np.array(pixels)

    if ra_range is not None or dec_range is not None:
        lon, lat = hp.pix2ang(nside, pixels, lonlat=True)
        if ra_range is not None:
            ra_mask = np.logical_and(ra_range[0] <= lon, lon <= ra_range[1])
        else:
            ra_mask = np.ones_like(lon, dtype=bool)
        if dec_range is not None:
            dec_mask = np.logical_and(dec_range[0] <= lat, lat <= dec_range[1])
        else:
            dec_mask = np.ones_like(lat, dtype=bool)

        mask  = ra_mask & dec_mask

        pixels = pixels[mask]

        if values is not None:
            try:
                # Convert to array if needed, but don't copy if already an array
                values_arr = np.asarray(values)
                if values_arr.ndim == 1:
                    values = values_arr[mask]
                elif values_arr.ndim == 2:
                    if freq_first:
                        values = values_arr[:, mask]
                    else:
                        values = values_arr[mask,:]
                else:
                    raise ValueError(f"values must be 1D or 2D. Got {values_arr.ndim}D")
            except (ValueError, TypeError, IndexError) as e:
                raise ValueError(f"values must be an array-like object. Got {type(values)}") from e
            
            return pixels, values

    return pixels

def build_hpmap(
    nside: int, pixels: np.ndarray, values: np.ndarray
) -> np.ma.MaskedArray:
    """
    Construct a masked Healpix map from pixel indices and values.

    Parameters
    ----------
    nside : int
        Healpix nside resolution parameter.
    pixels : np.ndarray
        Array of pixel indices.
    values : np.ndarray
        Array of values corresponding to the pixel indices.
    Returns
    -------
    healpix_map : hp.ma
        A masked Healpix map with unseen values set to hp.UNSEEN.
    """
    npix = hp.nside2npix(nside)
    full_map = np.full(npix, hp.UNSEEN, dtype=float)
    full_map[pixels] = values
    return hp.ma(full_map)


def extract_hpmap_slice(
    map_path: str,
    index: int = 98,
    key_value: Union[str, Iterable[str]] = "cleanmap",
    key_pix: str = "map_pix",
    key_freq: str = "freq",
    key_nside: str = "nside",
    freq_first: bool = True,
):
    """
    Extract a frequency slice from an HDF5 dataset and return as a Healpix map.

    Parameters
    ----------
    map_path : str
        Path to the HDF5 file.
    index : int, optional
        Frequency index to extract (default: 98).
    key_value : str or Iterable[str], optional
        Dataset key(s) for map values. Can be a single string or a list of strings
        (default: "cleanmap").
    key_pix : str, optional
        Dataset key for pixel indices (default: "map_pix").
    key_freq : str, optional
        Dataset key for frequency bins (default: "freq").
    key_nside : str, optional
        Dataset key for Healpix nside (default: "nside").

    Returns
    -------
    healpix_map : hp.ma or dict[str, hp.ma]
        Healpix masked map(s). If `key_value` is a list, returns a dict of maps.
    map_pix : np.ndarray
        Pixel indices of the map.
    freq_bins : np.ndarray
        Frequency bin array.
    nside : int
        Healpix nside resolution.
    """
    skip_freq = False
    with h5.File(map_path, "r") as f:
        if isinstance(key_value, str) and key_value not in f:
            raise KeyError(f"Dataset key '{key_value}' not found in file.")
        if isinstance(key_value, Iterable):
            for key in key_value:
                if key not in f:
                    raise KeyError(f"Dataset key '{key}' not found in file.")
        if key_pix not in f:
            raise KeyError(f"Dataset key '{key_pix}' not found in file.")
        if key_nside not in f:
            print(
                "Dataset key 'nside' not found in file. Trying to extract from attributes."
            )
            try:
                nside = f.attrs[key_nside]
            except KeyError:
                raise KeyError(f"Attribute '{key_nside}' not found in file.")
        if key_freq not in f:
            print(
                "Dataset key 'freq' not found in file. Skipping frequency extraction."
            )
            skip_freq = True

        nside = int(f[key_nside][()])  # type: ignore
        map_pix = f[key_pix][()]  # type: ignore
        if not skip_freq:
            freq_bins = f[key_freq][()]  # type: ignore
            print(
                f"Loaded map with nside={nside}, {len(map_pix)} pixels, {len(freq_bins)} frequency bins."  # type: ignore
            )
        else:
            freq_bins = None
            print(
                f"Loaded map with nside={nside}, {len(map_pix)} pixels, frequency extraction skipped."  # type: ignore
            )

        def _extract_slice(dataset: np.ndarray) -> np.ndarray:
            """Determine which axis corresponds to frequency and slice accordingly."""
            if dataset.ndim == 1:
                return dataset
            elif freq_first:
                return dataset[index]
            else:
                return dataset[:, index]

        if isinstance(key_value, str):
            healpix_map = build_hpmap(
                nside, map_pix, _extract_slice(f[key_value]) # type: ignore
            )

        elif isinstance(key_value, (list, tuple)):
            healpix_map = {
                key: build_hpmap(nside, map_pix, _extract_slice(f[key]))  # type: ignore
                for key in key_value
            }

        else:
            raise TypeError("key_value must be a string or list of strings")

    if not skip_freq:
        curr_freq = freq_bins[index]  # type: ignore
        print(f"Extracted map slice(s) at {curr_freq:.2f} MHz")

        return healpix_map, map_pix, freq_bins, nside
    else:
        return healpix_map, map_pix, nside


def get_map_center_range(
    nside: int, map_pix: np.ndarray, nest: bool = False, lat_offset: List[float] = [-5, 15], lon_offset: List[float] = [0, 0]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute the approximate center and coordinate ranges of a Healpix map.

    Parameters
    ----------
    nside : int
        Healpix nside.
    map_pix : np.ndarray
        Array of pixel indices.
    nest : bool, optional
        Whether the pixel ordering is NESTED (default: False, RING).

    Returns
    -------
    rot : list[float]
        Center rotation for Healpy [lon_center, lat_center, 0].
    lonra : list[float]
        Longitude range [lon_min, lon_max] in degrees.
    latra : list[float]
        Latitude range [lat_min, lat_max] in degrees.
    """
    # Convert pixel indices to spherical coordinates
    theta, phi = hp.pix2ang(nside, map_pix, nest=nest)
    lon = np.rad2deg(phi)
    lat = 90.0 - np.rad2deg(theta)

    # --- Compute center ---
    lon_center = np.rad2deg(
        np.arctan2(np.mean(np.sin(np.deg2rad(lon))), np.mean(np.cos(np.deg2rad(lon))))
    )
    if lon_center < 0:
        lon_center += 360
    lat_center = np.mean(lat)

    # --- Compute ranges ---
    # Handle wrap-around at 0/360°
    lon_sorted = np.sort(lon)
    lon_diff = np.diff(np.concatenate([lon_sorted, [lon_sorted[0] + 360]]))
    max_gap_idx = np.argmax(lon_diff)
    lon_min = lon_sorted[(max_gap_idx + 1) % len(lon_sorted)]
    lon_max = lon_sorted[max_gap_idx]

    lat_min = lat.min()
    lat_max = lat.max()

    # lonra = [lon_min, lon_max]
    # latra = [lat_min, lat_max]
    lonra = [lon_min - lon_center + lon_offset[0], lon_max - lon_center + lon_offset[1]]
    latra = [lat_min - lat_center + lat_offset[0], lat_max - lat_center + lat_offset[1]]
    rot = [lon_center, lat_center, 0]

    return rot, lonra, latra

def shift_pixel_to_target(nside, src_pix, original_field=(0, 0), new_field=(90, 0)):
    """
    Shift the spherical coordinates of the source pixel to a new field center.

    :param nside: HEALPix nside parameter.
    :param src_pix: Source pixel index.
    :param original_field: Tuple (longitude, latitude) of the original field center, default (0, 0) degrees.
    :param new_field: Tuple (longitude, latitude) of the new field center, default (90, 0) degrees.
    :return: New pixel index after shifting.
    """
 
    src_lon, src_lat = hp.pix2ang(nside, src_pix, lonlat=True)
    
    # 计算相对偏移（平移量）
    delta_lon =  original_field[0] - new_field[0]
    delta_lat =  original_field[1] - new_field[1]
    
    new_lon = src_lon - delta_lon
    new_lat = src_lat - delta_lat
    # 确保新的经纬度在有效范围内, # 经度范围 [0, 360), 纬度范围 [-90, 90)
    # print(new_lon, new_lat)
    new_lon = new_lon % 360
    new_lat[new_lat > 90]  -= 180
    new_lat[new_lat < -90] += 180
    # print(new_lon, new_lat)
    # 将新的球面坐标转换为像素索引
    new_pix = hp.ang2pix(nside, new_lon, new_lat, lonlat=True)
    
    return new_pix

def find_pixels_within_radius(nside, ra, dec, radius_deg):
    """
    在指定半径内找到 HEALPix 像素。

    此函数使用 `healpy.query_disc` 高效地找到从指定天空点开始、在给定角半径内的所有像素索引。

    参数:
    - nside (int): HEALPix nside 参数，确定分辨率。
    - ra (float): 中心点的赤经（RA），单位为度。
    - dec (float): 中心点的赤纬（Dec），单位为度。
    - radius_deg (float): 角半径，单位为度。

    返回:
    - numpy.ndarray: 在指定半径内的像素索引数组。
    """
    return hp.query_disc(
        nside, hp.ang2vec(ra, dec, lonlat=True), np.deg2rad(radius_deg), inclusive=True
    )

def load_stack_map(
    path: str,
    unit_factor: float = 1,
    key_pix: str = "stack_pix",
    key_val: str = "stack_signal",
    key_nside: str = "nside",
) -> np.ma.MaskedArray:
    """
    Load galaxy stacking results from an HDF5 file and return as a Healpix map.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    unit_factor : float, optional
        Factor to scale the map values (default: 1).
    key_pix : str, optional
        Dataset key for Healpix pixel indices (default: "stack_pix").
    key_val : str, optional
        Dataset key for stacking signal values (default: "stack_signal").
    key_nside : str, optional
        Dataset key for the nside value (default: "nside").

    Returns
    -------
    hp.ma
        A masked Healpix map containing the stacked galaxy signal.
    """
    # pixels, values, nside = read_h5(path, [key_pix, key_val, key_nside])
    with h5.File(path, "r") as f:
        nside = int(f[key_nside][()]) # type: ignore
        pixels = f[key_pix][()] # type: ignore
        values = f[key_val][()] # type: ignore
        return build_hpmap(nside, pixels, values) * unit_factor # type: ignore