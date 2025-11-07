"""
A script to shuffle the frequency slices of cuboid map, and also copy the rest datasets.
"""

import os
import sys
from typing import Any, List, Optional

import h5py
import numpy as np

# --- 1. Helper Functions: Attribute and Data Copying ---


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


# --- 2. Main Shuffle and Copy Logic ---


def process_shuffle(
    input_filepath: str,
    output_filepath: str,
    shuffle_keys: List[str],
    copy_keys: List[str],
    shuffle_axis: int = 0,
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

    # Set for fast lookup and checking if already processed
    all_processed_keys = set()

    try:
        # Open input file and create/overwrite output file
        with (
            h5py.File(input_filepath, "r") as fin,
            h5py.File(output_filepath, "w") as fout,
        ):
            # 1. Process root attributes
            copy_attributes(fin, fout)

            # 2. Execute Shuffle
            if shuffle_keys:
                print("\n--- Executing Shuffle ---")

                # Get a dataset, read its size, and generate shuffle indices
                first_dset_key = shuffle_keys[0]
                first_dset = get_dataset(fin, first_dset_key)

                if (
                    first_dset is None
                    or not isinstance(first_dset, h5py.Dataset)
                    or first_dset.ndim < 3
                ):
                    print(
                        f"Error: First shuffle dataset '{first_dset_key}' is not a valid 3D dataset."
                    )
                    return

                num_slices = first_dset.shape[shuffle_axis]
                shuffle_indices = np.arange(num_slices)
                np.random.shuffle(shuffle_indices)
                print(f"Generated {num_slices} shuffle indices on axis {shuffle_axis}.")

                # Apply the same indices to all datasets that need to be shuffled
                for dset_key in shuffle_keys:
                    dset_in = get_dataset(fin, dset_key)
                    if dset_in is None or not isinstance(dset_in, h5py.Dataset):
                        print(
                            f"Warning: Dataset '{dset_key}' not found or is not a Dataset. Skipping."
                        )
                        continue

                    if dset_in.ndim < 3:
                        print(f"Warning: Dataset '{dset_key}' is not 3D. Skipping.")
                        continue

                    print(f"Shuffling dataset: {dset_key} (Shape: {dset_in.shape})")

                    # Read the entire dataset into memory (Note: for very large datasets, chunked reading/writing may be necessary)
                    data = dset_in[:]

                    # Create a slice object to apply the shuffle indices
                    # For example, if shuffle_axis=0, the slice is [shuffle_indices, :, :]
                    slices = [slice(None)] * dset_in.ndim
                    slices[shuffle_axis] = shuffle_indices

                    shuffled_data = data[tuple(slices)]

                    # Create and write the shuffled dataset in the output file
                    dset_out = fout.create_dataset(
                        dset_key, data=shuffled_data, compression="gzip"
                    )
                    copy_attributes(dset_in, dset_out)
                    all_processed_keys.add(dset_key.strip("/"))

            # 3. Copy Specified Data (Groups and Datasets)
            if copy_keys:
                print("\n--- Executing Direct Copy ---")
                for key in copy_keys:
                    path = key.strip("/")
                    if path in all_processed_keys:
                        print(
                            f"Warning: Item '{key}' already processed (shuffled). Skipping copy."
                        )
                        continue

                    print(f"Copying item: {key}")

                    # Recursively copy Group or Dataset
                    copy_h5_item_recursive(fin, fout, path)
                    all_processed_keys.add(path)

            # 4. Final Check (Optional: can iterate over remaining unprocessed objects and copy them)
            # In this simplified version, we only process explicitly specified keys

            print(f"\nSuccessfully processed file and saved to {output_filepath}")

    except Exception as e:
        print(f"Fatal Error during HDF5 processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- 3. Script Entry Point ---

    # Read configuration from environment variables
    # Note: In Linux/macOS, list elements are usually separated by commas
    INPUT_FILE = os.environ.get("INPUT_PATH")
    OUTPUT_FILE = os.environ.get("OUTPUT_PATH")
    SHUFFLE_KEYS_STR = os.environ.get("SHUFFLE_KEYS", "")
    COPY_KEYS_STR = os.environ.get("COPY_KEYS", "")
    SHUFFLE_AXIS = int(os.environ.get("SHUFFLE_AXIS", "0"))

    # Validate input
    if not INPUT_FILE or not OUTPUT_FILE:
        print("Error: INPUT_PATH and OUTPUT_PATH environment variables must be set.")
        sys.exit(1)

    # Convert keys string to list
    shuffle_keys = [k.strip() for k in SHUFFLE_KEYS_STR.split(",") if k.strip()]
    copy_keys = [k.strip() for k in COPY_KEYS_STR.split(",") if k.strip()]

    if not shuffle_keys and not copy_keys:
        print(
            "Warning: Neither SHUFFLE_KEYS nor COPY_KEYS are provided. Nothing will be processed."
        )
        # exit
        sys.exit(0)

    # Execute processing
    # Assuming your 3D data is [time/slice, width, height], so the first dimension (0) is shuffled by default
    process_shuffle(
        input_filepath=INPUT_FILE,
        output_filepath=OUTPUT_FILE,
        shuffle_keys=shuffle_keys,
        copy_keys=copy_keys,
        shuffle_axis=SHUFFLE_AXIS,
    )
