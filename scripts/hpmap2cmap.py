"""
A script to project the HEALPix map into a cuboid one using `MollweideSkyproj` from `skyproj`.
"""

import multiprocessing as mp
import os

import h5py as h5
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj


def check_file_extension(filename: str, extension: str = ".h5") -> str:
    """Check if the filename has the given extension, and add it if not."""
    if not filename.endswith(extension):
        filename += extension
    return filename


def get_pix_info(map_pix, nside=2048):
    lon, lat = hp.pix2ang(nside, map_pix, lonlat=True)

    dlon = lon.max() - lon.min()
    dlat = lat.max() - lat.min()
    lon0 = dlon * 0.5 + lon.min()
    lat0 = dlat * 0.5 + lat.min()

    ratio = dlat / dlon

    return lon0, lat0, ratio


def project_healpix(
    nside: int,
    pixels,
    values,
    lon0: float = 0,
    xsize=200,
    aspect=1.0,
    return_raster: bool = False,
    show_plot: bool = False,
    fig_args: dict = {"figsize": (6, 4)},
    sp_args: dict = {"celestial": False},
    draw_args: dict = {"cmap": "bwr", "norm": "linear"},
    cbar_args: dict = {"label": "K"},
) -> None | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the Healpix data to 2d array using skyproj.

    Args:
        nside (int): The NSIDE parameter of the Healpix map.
        pixels (np.ndarray): Array of HEALPix pixel indices you have data for. Dimension: (n,).
        values (np.ndarray): Array of corresponding data values for each pixel. Dimension: (n,).
        lon0 (float): The longitude of the central meridian of the projection. Defaults to 0.

        return_raster (bool): If True, returns the rasterized longitude, latitude, and values. Defaults to False.
        show_plot (bool): If True, displays the 2D projection. Defaults to False.
        fig_args (dict): Additional arguments for creating the figure. Defaults to {'figsize':(6,4)}.
        sp_args (dict): Additional arguments for creating the Skyproj object. Defaults to {'celestial':False}.
        draw_args (dict): Additional arguments for drawing the data. Defaults to {'cmap':'bwr', 'norm':'linear'}.
        cbar_args (dict): Additional arguments for the colorbar. Defaults to {'label':'K'}.
    Returns:
        None or tuple: If return_raster is True, returns (lon_raster, lat_raster, values_raster).
                       Otherwise, returns None.
    """
    lon_raster, lat_raster, values_raster = None, None, None

    if show_plot:
        _, ax = plt.subplots(**fig_args)
        sp = skyproj.MollweideSkyproj(ax=ax, lon_0=lon0, **sp_args)
        # Ensure data is not empty before drawing
        if len(pixels) > 0 and len(values) > 0:
            _, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(
                nside, pixels, values, xsize=xsize, aspect=aspect, **draw_args
            )

        sp.draw_inset_colorbar(**cbar_args)
        sp.ax.set_ylabel("Dec.", fontsize="large")
        sp.ax.set_xlabel("R.A.", fontsize="large")
        plt.show()

    else:
        temp_fig, ax = plt.subplots(figsize=(1, 1))  # Minimal figure size
        sp = skyproj.MollweideSkyproj(ax=ax, lon_0=lon0, **sp_args)

        # Ensure data is not empty before drawing
        if len(pixels) > 0 and len(values) > 0:
            _, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(
                nside, pixels, values, xsize=xsize, aspect=aspect, **draw_args
            )
        plt.close(temp_fig)  # Close the temporary figure to free memory

    if return_raster:
        if lon_raster is None or lat_raster is None or values_raster is None:
            raise ValueError("No data was projected; check input pixels and values.")
        return lon_raster, lat_raster, values_raster


def _project_one_slice(args) -> np.ndarray:
    """
    Helper function to project data for a single frequency slice.
    Designed to be used with multiprocessing.Pool.map.
    """
    nside, pixels, values_one_slice, lon0, xsize, aspect, kwargs = args
    # project_healpix handles the temporary figure creation and closing
    result = project_healpix(
        nside,
        pixels,
        values_one_slice,
        lon0=lon0,
        xsize=xsize,
        aspect=aspect,
        return_raster=True,
        **kwargs,
    )
    if result is None:
        raise ValueError(
            "project_healpix returned None; check input pixels and values."
        )
    _, _, values_raster = result
    return (
        values_raster  # Only return values_raster for efficiency in parallel processing
    )


def project_healpix_3d(
    nside: int,
    pixels: np.ndarray,
    values: np.ndarray,
    lon0: float = 0,
    xsize: int = 200,
    aspect: float = 1.0,
    nworker: int | None = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the Healpix data to 3d array (with frequency as first dimension) using skyproj,
    with parallel processing for each frequency band.

    Args:
        nside (int): The NSIDE parameter of the Healpix map.
        pixels (np.ndarray): Array of HEALPix pixel indices you have data for. Dimension: (n,).
        data (np.ndarray): Array of corresponding data values for each pixel. Dimension: (m, n).
        lon0 (float): The longitude of the central meridian of the projection. Defaults to 0.
        xsize (int): The width of the output raster in pixels. Defaults to 200.
        aspect (float): The aspect ratio of the output raster. The height will be calculated as xsize*aspect. Defaults to 1.
        nworker (int): Number of parallel processes to use. Defaults to 1 to not use multiprocessing. Set to None to use all available cores.
        **kwargs: Additional arguments pass to `project_healpix` (e.g., cmap, norm).

    Returns:
        tuple: (lon_raster, lat_raster, values_raster_3d).
               lon_raster and lat_raster are 2D numpy arrays.
               values_raster_3d is a 3D numpy array (num_frequencies, lat_res, lon_res).
    """
    if values.ndim != 2:
        raise ValueError(
            "Input 'data' must be a 2D array (num_frequencies, num_pixels)."
        )
    if values.shape[1] != len(pixels):
        raise ValueError(
            "Second dimension of 'data' must match the length of 'pixels'."
        )

    num_frequencies = values.shape[0]

    # Prepare arguments for task
    args_list = [
        (nside, pixels, values[i], lon0, xsize, aspect, kwargs)
        for i in range(num_frequencies)
    ]

    # if nworker is None, use all available cores:
    nworker = mp.cpu_count() if nworker is None else nworker

    # if nworker is 1, use the sequential version:
    if nworker == 1:
        raster_values_list = [_project_one_slice(args) for args in args_list]

        print(f"Starting sequential projection for {num_frequencies} frequencies...")

    # else, ensure at least two process for multiprocessing
    else:
        nworker = max(
            2, min(nworker, num_frequencies)
        )  # Limit to number of frequencies

        print(
            f"Starting parallel projection for {num_frequencies} frequencies using {nworker} processes..."
        )

        # Use multiprocessing.Pool to parallelize the projection
        with mp.Pool(processes=nworker) as pool:
            # pool.map returns results in the same order as inputs
            raster_values_list = pool.map(_project_one_slice, args_list)

    # Get lon_raster and lat_raster from the first projection (they should be the same for all)
    result = project_healpix(
        nside,
        pixels,
        values[0],
        lon0=lon0,
        xsize=xsize,
        aspect=aspect,
        return_raster=True,
        **kwargs,
    )
    if result is None:
        raise ValueError(
            "project_healpix returned None; check input pixels and values."
        )
    lon_raster, lat_raster, _ = result

    return lon_raster, lat_raster, np.array(raster_values_list)


def _project_and_save(
    nside: int,
    pixels: np.ndarray,
    values: np.ndarray,
    output_file: str,
    xsize: int = 200,
    nworker: int | None = 1,
    **kwargs,
) -> None:
    """
    Project the data for a single tile onto a 3D array and save it to the output HDF5 file.

    Args:
        nside (int): The NSIDE parameter of the Healpix map.
        output_file (str): Path to the output HDF5 file.
        base_path (str): Base path for the output file. If None, uses the current directory.
        xsize (int): The width of the output raster in pixels. Defaults to 200.
        nworker (int): Number of parallel processes to use. Defaults to 1 to not use multiprocessing. Set to None to use all available cores.
        **kwargs: Additional arguments pass to `project_healpix`.
    """

    # load the tile data
    # lon_range, _ = get_lon_lat_range(nside, pixels)

    lon0, lat0, ratio = get_pix_info(pixels, nside)

    # Project the data onto a 3D array
    lon_raster, lat_raster, values_raster = project_healpix_3d(
        nside,
        pixels,
        values,
        lon0,
        xsize,
        ratio,
        nworker,
        **kwargs,
    )

    # check output file
    output_file = check_file_extension(output_file, ".h5")

    # Save the projected data to the output HDF5 file
    with h5.File(output_file, "a") as f_out:
        f_out.create_dataset(
            name="lon_raster", data=lon_raster, dtype=DTYPE_D, compression="gzip"
        )
        f_out.create_dataset(
            name="lat_raster", data=lat_raster, dtype=DTYPE_D, compression="gzip"
        )
        f_out.create_dataset(
            name="values_raster", data=values_raster, dtype=DTYPE_D, compression="gzip"
        )

    print(f"Saved the projected data to {output_file}")


def _do_project(
    data_path: str,
    key_map: str,
    key_pix: str,
    output_prefix: str | None = None,
    output_base: str | None = None,
    xsize: int = 200,
    nworker: int | None = 1,
    copy_data_keys: list[str] | None = None,
    **kwargs,
) -> None:
    """
    Splits the Healpix data into tiles, projects each tile onto a 3D array, and saves the projected data to an HDF5 file.

    Args:
        data_path (str): Path to the HDF5 file containing the Healpix data.
        key_map (str): The name of the dataset in the HDF5 file containing the map values.
        key_pix (str): The name of the dataset in the HDF5 file containing the pixel indices.
        output_prefix (str): The prefix for the output HDF5 files containing the split data. Defaults to None, use 'projected_nside{nside}_xsize{xsize}.h5'.
        output_base (str | None): Base path for the output file. If None, uses the current directory.
        xsize (int): The width of the output raster in pixels. Defaults to 200.
        nworker (int): Number of parallel processes to use. Defaults to 1 to not use multiprocessing. Set to None to use all available cores.
        copy_data_keys (list[str] | None): List of keys to copy from the input file to the output file. Defaults to None.
        **kwargs: Additional arguments to pass to the `do_projection` function.
    """

    with h5.File(data_path, "r") as f_in:
        values = f_in[key_map][:]  # type: ignore
        pixels = f_in[key_pix][:]  # type: ignore
        nside = f_in["nside"][()]  # type: ignore
        copy_data = (
            {k: f_in[k][:] for k in copy_data_keys}
            if copy_data_keys is not None
            else {}
        )  # type: ignore

    output_prefix = (
        f"projected_nside{nside}_xsize{xsize}"
        if output_prefix is None
        else output_prefix
    )
    output_base = os.getcwd() if output_base is None else output_base

    output_file = os.path.join(output_base, output_prefix + ".h5")

    _project_and_save(nside, pixels, values, output_file, xsize, nworker, **kwargs)  # type: ignore

    for k, v in copy_data.items():
        with h5.File(output_file, "a") as f_out:
            f_out.create_dataset(name=k, data=v)

    return None


def load_projected_data(
    file_path,
    freq: int | list | slice | None = 0,
    keys: list[str] = ["lon_raster", "lat_raster", "values_raster"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the projected data from the HDF5 file.
    Args:
        file_path (str): Path to the HDF5 file containing the projected data.
        freq (int | list | slice | None): Frequency index or indices to load. If None, loads all frequencies.
    Returns:
        tuple: A tuple containing:
            - lon (np.ndarray): 2D array of longitudes.
            - lat (np.ndarray): 2D array of latitudes.
            - data (np.ndarray): 3D array of projected values (frequencies, latitudes, longitudes).
    """

    with h5.File(file_path, "r") as fin:
        # load the lon, lat and data
        lon_key, lat_key, data_key = keys
        lon = np.array(fin[lon_key][()])  # type: ignore
        lat = np.array(fin[lat_key][()])  # type: ignore

        # if freq is int or list or slice, load the corresponding data
        if (
            isinstance(freq, int)
            or (isinstance(freq, list) and all(isinstance(f, int) for f in freq))
            or isinstance(freq, slice)
        ):
            data = np.array(fin[data_key][freq])  # type: ignore
        # if freq is None, load all data
        elif freq is None:
            data = np.array(fin[data_key][()])  # type: ignore
        # if freq is not valid, raise ValueError
        else:
            raise ValueError(f"Invalid freq value: {freq}")

    return lon, lat, data


if __name__ == "__main__":
    # Read environment variables

    ## necessary environment variables
    DATA_PATH = os.getenv("DATA_PATH")  # the path of the healpix data
    KEY_MAP = os.getenv("KEY_MAP")  # the key of the data in the healpix data
    KEY_PIX = os.getenv("KEY_PIX")  # the key of the pixels in the healpix data

    if not DATA_PATH or not KEY_MAP or not KEY_PIX:
        raise ValueError(
            "Please set the environment variables NCORE, DATA_PATH, KEY_DATA, KEY_PIXELS"
        )

    ## optional environment variables
    global DTYPE_D, DTYPE_P

    ## do not accept None as default value
    XSIZE = int(os.getenv("XSIZE", 200))
    DTYPE_D = os.getenv("DTYPE_D", "f4")
    DTYPE_P = os.getenv("DTYPE_P", "i8")

    ## accept None as default value
    NCORE = os.getenv("NCORE")
    OUTPUT_BASE = os.getenv("OUTPUT_BASE")
    OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX")
    COPY_DATA_KEYS = os.getenv("COPY_DATA_KEYS")

    ## remind the default values of optional environment variables
    if NCORE is not None:
        NCORE = int(NCORE)
    else:
        print(
            "NCORE is not set, use ALL cpu cores. If you want to disable multiprocessing, set NCORE to 1.\n"
        )
    if not OUTPUT_BASE:
        print("OUTPUT_BASE is not set, output in the current directory\n")
    if not OUTPUT_PREFIX:
        print(
            "PROJECT_OUTPUT_PREFIX is not set, use default value 'projected_nside{nside}_xsize{xsize}.h5'\n"
        )
    if not DTYPE_D:
        print("DTYPE_D is not set, use default value 'f4'\n")
    if not DTYPE_P:
        print("DTYPE_P is not set, use default value 'i8'\n")
    if COPY_DATA_KEYS is not None:
        COPY_DATA_KEYS = COPY_DATA_KEYS.split(",")
        print(f"Copy data keys: {COPY_DATA_KEYS}\n")
    else:
        print("COPY_DATA_KEYS is not set, do not copy any data\n")

    # do the splite and project
    _do_project(
        DATA_PATH,
        KEY_MAP,
        KEY_PIX,
        OUTPUT_PREFIX,
        OUTPUT_BASE,
        XSIZE,
        NCORE,
        COPY_DATA_KEYS,
    )
