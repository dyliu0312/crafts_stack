"""
A script to select pixels of simulation build HEALPix map to mock mock observations (CRAFTS HIIM map).
"""
import os
import h5py as h5
from crafts_stack.hpmap import shift_pixel_to_target
from mytools.data import read_h5, save_h5, save_attrs

def find_positions(large_list, subset_list):
    """
    Finds the last occurrence index of each element from a subset list within a larger list.
    
    Args:
        large_list (list): The list to search within.
        subset_list (list): The list of elements to find.
        
    Returns:
        list: A list of indices corresponding to the positions of elements in subset_list.
              Returns None for any element not found.
    """
    index_map = {}
    # Create a mapping of each value in large_list to its last index for quick lookup.
    for idx, value in enumerate(large_list):
        index_map[value] = idx
    # Use the map to find the index for each element in subset_list.
    return [index_map.get(value, None) for value in subset_list]

def get_flag(nside, target_map_pix, original_map_pix, target_filed_center, original_filed_center):
    """
    Generates a flag (list of indices) to crop the original map.
    
    This function first shifts the pixel indices from the target map's
    coordinate system to the original map's system, then finds their
    corresponding indices in the original map's pixel list.
    
    Args:
        nside (int): The HEALPix resolution parameter.
        target_map_pix (np.array): Pixel indices of the target map.
        original_map_pix (np.array): Pixel indices of the original map.
        target_filed_center (list): Center coordinates of the target field.
        original_filed_center (list): Center coordinates of the original field.
        
    Returns:
        list: A list of indices (the "flag") to be used for slicing the original data.
    """
    # Shift target pixel indices to align with the original map's field center.
    map_pix_shifted = shift_pixel_to_target(nside, target_map_pix, target_filed_center, original_filed_center)
    # Find the indices of the shifted pixels in the original map's pixel list.
    flag = find_positions(original_map_pix, map_pix_shifted)
    return flag

def cut_save_dset(
        output_file, flag, 
        original_file, original_dset_key_map, original_dset_key_freq, 
        target_map_pix, target_filed_center, original_filed_center
):
    """
    Cuts a dataset from the original file and saves it to a new file with metadata.
    
    Args:
        output_file (str): The path to the output HDF5 file.
        flag (list): The list of indices for slicing the original data.
        original_file (str): The path to the original HDF5 file.
        original_dset_key_map (str): The key for the map dataset in the original file.
        original_dset_key_freq (str): The key for the frequency dataset in the original file.
        target_map_pix (np.array): Pixel indices of the target map.
        target_filed_center (list): Center coordinates of the target field.
        original_filed_center (list): Center coordinates of the original field.
    """
    # Read the map and frequency data from the original HDF5 file.
    dset_map, dset_freq = read_h5(original_file, [original_dset_key_map, original_dset_key_freq])
    # Use the 'flag' to slice the map data, effectively cutting it.
    dset_map_cut = dset_map[:,flag]
    # Save the cut map, target pixel indices, and frequency data to the new HDF5 file.
    save_h5(output_file, ['map_value', 'map_pix', original_dset_key_freq], [dset_map_cut, target_map_pix, dset_freq])

    # Prepare attributes for the new HDF5 file.
    attributes = {
        "new_filed_center": target_filed_center,
        "original_filed_center": original_filed_center,
    }

    # Open the original file to copy its attributes.
    with h5.File(original_file, 'r') as f:
        original_attrs = f.attrs.items()
        attributes.update(original_attrs)
    
    # Save all collected attributes to the new HDF5 file.
    save_attrs(output_file, attributes)

def _get_env_var():
    """
    A helper function to read all necessary configuration from environment variables.
    
    Returns:
        dict: A dictionary of all configuration arguments.
    """
    # Read HEALPix resolution, file paths, and dataset keys from environment variables.
    nside = int(os.getenv("NSIDE", '256'))

    target_file = os.getenv("TARGET_FILE")
    target_pixel_key = os.getenv("TARGET_PIX_KEY", "map_pix")
    target_filed_center_str = os.getenv("TARGET_FILED_CENTER", '218,42')
    target_filed_center = [float(i) for i in target_filed_center_str.split(",")] # type: ignore

    original_file = os.getenv("ORIGINAL_FILE")
    original_pixel_key = os.getenv("ORIGINAL_PIX_KEY", "map_pix")
    original_filed_center_str = os.getenv("ORIGINAL_FILED_CENTER", '90,0')
    original_filed_center = [float(i) for i in original_filed_center_str.split(",")] # type: ignore

    original_dset_key_map = os.getenv("ORIGINAL_DSET_KEY_MAP", "clean_map")
    original_dset_key_freq = os.getenv("ORIGINAL_DSET_KEY_FREQ", "freq")

    output_file = os.getenv("OUTPUT_FILE", "./mock_healpix_map.h5")

    # Store all arguments in a dictionary.
    args = {
        "nside": nside,
        "target_file": target_file,
        "target_pixel_key": target_pixel_key,
        "target_filed_center": target_filed_center,
        "original_file": original_file,
        "original_pixel_key": original_pixel_key,
        "original_filed_center": original_filed_center,
        "original_dset_key_map": original_dset_key_map,
        "original_dset_key_freq": original_dset_key_freq,
        "output_file": output_file,
    }
    return args

def main():
    """
    The main function to orchestrate the entire process.
    """
    # 1. Get all configuration settings.
    args = _get_env_var()

    # 2. Read the pixel indices from both target and original files.
    target_map_pix = read_h5(args["target_file"], args["target_pixel_key"])
    original_map_pix = read_h5(args["original_file"], args["original_pixel_key"])
    
    # 3. Generate the flag (list of indices) for cutting the data.
    flag = get_flag(args['nside'], target_map_pix, original_map_pix, args['target_filed_center'], args['original_filed_center'])

    # 4. Perform the data cutting and saving operation.
    cut_save_dset(
        args["output_file"], flag,
        args["original_file"], args["original_dset_key_map"], args["original_dset_key_freq"],
        target_map_pix, args["target_filed_center"], args["original_filed_center"],
    )

    # 5. Print a success message.
    print(f"✅ Mock healpix map saved to {args['output_file']}")

if __name__ == "__main__":
    # Ensure the main function is executed only when the script is run directly.
    main()
