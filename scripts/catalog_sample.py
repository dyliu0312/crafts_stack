#!/usr/bin/env python3
"""
Fast HDF5 Galaxy Catalog Sampler.
Optimized: Loads individual datasets into RAM for fast slicing, then flushes to disk.
"""

import argparse
import os
import sys
import time
from typing import Optional

import h5py
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDF5 Galaxy Catalog Sampler (Fast RAM Mode)"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=os.getenv("CATALOG_INPUT"),
        help="Input HDF5 file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.getenv("CATALOG_OUTPUT"),
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=os.getenv("N_SAMPLES"),
        help="Number of samples to extract",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=os.getenv("RANDOM_SEED"), help="Random seed"
    )

    args = parser.parse_args()
    if not all([args.input, args.output, args.samples]):
        parser.error("Arguments --input, --output, and --samples are required.")
    return args


def get_primary_dimension(h5_file: h5py.File) -> int:
    """Finds the most common first-dimension size (galaxy count)."""
    shapes = []

    def collect_shapes(name, node):
        if isinstance(node, h5py.Dataset) and node.shape:
            shapes.append(node.shape[0])

    h5_file.visititems(collect_shapes)
    if not shapes:
        raise ValueError("No datasets found.")
    return int(np.argmax(np.bincount(shapes)))


def copy_attributes(source, dest):
    """Copy HDF5 attributes/metadata."""
    for k, v in source.attrs.items():
        dest.attrs[k] = v


def process_sampling(
    input_path: str, output_path: str, n_samples: int, seed: Optional[int]
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Performance timing
    start_time = time.time()

    with h5py.File(input_path, "r") as f_in:
        total_galaxies = get_primary_dimension(f_in)

        # Generate Indices
        if n_samples >= total_galaxies:
            print(f"Request {n_samples} >= Total {total_galaxies}. Copying all.")
            indices = slice(None)  # Slice all
            n_samples = total_galaxies
        else:
            if seed is not None:
                np.random.seed(seed)
            # Sorting indices is strictly faster for memory access
            indices = np.sort(
                np.random.choice(total_galaxies, size=n_samples, replace=False)
            )

        print(f"Input: {input_path}")
        print(f"Sampling {n_samples} / {total_galaxies} items")

        with h5py.File(output_path, "w") as f_out:
            copy_attributes(f_in, f_out)

            # We collect dataset names first to allow for a simple progress counter
            dataset_names = []
            f_in.visititems(
                lambda name, node: dataset_names.append(name)
                if isinstance(node, h5py.Dataset)
                else None
            )
            total_dsets = len(dataset_names)

            print(f"Processing {total_dsets} datasets...")

            count = 0

            def visitor_func(name, node):
                nonlocal count

                # Handle Groups
                if isinstance(node, h5py.Group):
                    if name not in f_out:
                        g = f_out.create_group(name)
                        copy_attributes(node, g)
                    return

                # Handle Datasets
                if isinstance(node, h5py.Dataset):
                    count += 1
                    # Simple progress indicator
                    if count % 10 == 0:
                        sys.stdout.write(
                            f"\rProgress: {count}/{total_dsets} datasets processed"
                        )
                        sys.stdout.flush()

                    # Logic: Is this a galaxy-array or metadata?
                    is_galaxy_array = node.shape and node.shape[0] == total_galaxies

                    # --- FAST RAM LOADING ---
                    # 1. Read the FULL dataset into memory (Sequential Read = Fast)
                    # 2. Slice it in memory (Instant)
                    if is_galaxy_array:
                        # The [:] forces a read into a numpy array in RAM
                        full_data = node[:]
                        data_to_write = full_data[indices]
                        del full_data  # Free RAM immediately
                    else:
                        # Static metadata, just copy
                        data_to_write = node[()]

                    # Write to output with compression
                    dset = f_out.create_dataset(
                        name, data=data_to_write, compression="gzip", compression_opts=4
                    )
                    copy_attributes(node, dset)

            f_in.visititems(visitor_func)

    print(f"\n\nDone! Saved to: {output_path}")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    try:
        args = parse_arguments()
        process_sampling(args.input, args.output, args.samples, args.seed)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
