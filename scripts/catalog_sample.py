#!/usr/bin/env python3
"""
Randomly sample galaxies from HDF5 catalog data while preserving data structure.
"""

import os
import h5py
import numpy as np
import sys
from typing import Dict, Any

def get_environment_variables() -> Dict[str, Any]:
    """
    Read configuration parameters from environment variables
    """
    config = {
        'input_file': os.getenv('CATALOG_INPUT'),
        'output_file': os.getenv('CATALOG_OUTPUT'),
        'random_seed': os.getenv('RANDOM_SEED'),
        'n_samples': os.getenv('N_SAMPLES')
    }
    
    # Validate required parameters
    if not config['input_file']:
        raise ValueError("Environment variable CATALOG_INPUT is not set")
    if not config['output_file']:
        raise ValueError("Environment variable CATALOG_OUTPUT is not set")
    if not config['n_samples']:
        raise ValueError("Environment variable N_SAMPLES is not set")
    
    try:
        config['n_samples'] = int(config['n_samples']) # type: ignore
    except ValueError:
        raise ValueError("Environment variable N_SAMPLES must be an integer")
    
    if config['random_seed']:
        try:
            config['random_seed'] = int(config['random_seed']) # type: ignore
        except ValueError:
            raise ValueError("Environment variable RANDOM_SEED must be an integer")
    
    return config

def read_h5_structure(h5_file: h5py.File) -> Dict[str, Any]:
    """
    Read HDF5 file structure information
    """
    structure = {}
    
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            structure[name] = {
                'shape': node.shape,
                'dtype': node.dtype,
                'size': node.size
            }
    
    h5_file.visititems(visit_func)
    return structure

def get_total_galaxies(h5_file: h5py.File) -> int:
    """
    Get total number of galaxies (assuming first dimension is galaxy count)
    """
    for name in h5_file.keys():
        if isinstance(h5_file[name], h5py.Dataset):
            return h5_file[name].shape[0] # type: ignore
    raise ValueError("Cannot determine total galaxies: no valid datasets found")

def random_sample_galaxies(h5_file: h5py.File, n_samples: int, random_seed: int = None) -> Dict[str, np.ndarray]:
    """
    Randomly sample galaxy data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_galaxies = get_total_galaxies(h5_file)
    
    if n_samples > total_galaxies:
        print(f"Warning: Requested sample size {n_samples} exceeds total galaxies {total_galaxies}")
        print(f"Will sample all {total_galaxies} galaxies")
        n_samples = total_galaxies
    
    # Generate random indices
    indices = np.random.choice(total_galaxies, size=n_samples, replace=False)
    indices.sort()  # Maintain original order
    
    sampled_data = {}
    
    # Traverse all datasets and sample data at corresponding indices
    def sample_datasets(name, node):
        if isinstance(node, h5py.Dataset):
            # Assume first dimension is galaxy index
            if node.shape[0] == total_galaxies:
                sampled_data[name] = node[indices]
            else:
                # If dimension doesn't match, copy entire dataset (e.g., constant data)
                sampled_data[name] = node[()]
    
    h5_file.visititems(sample_datasets)
    return sampled_data

def save_sampled_data(sampled_data: Dict[str, np.ndarray], output_file: str):
    """
    Save sampled data to new HDF5 file
    """
    with h5py.File(output_file, 'w') as f_out:
        for dataset_name, data in sampled_data.items():
            # Create dataset with same structure as original data
            f_out.create_dataset(dataset_name, data=data, compression='gzip')
    
    print(f"Sampled data saved to: {output_file}")

def list_datasets(input_file: str):
    """
    List all datasets in HDF5 file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    with h5py.File(input_file, 'r') as h5_in:
        structure = read_h5_structure(h5_in)
        total_galaxies = get_total_galaxies(h5_in)
        
        print(f"File: {input_file}")
        print(f"Total galaxies: {total_galaxies}")
        print("\nDataset list:")
        for name, info in structure.items():
            print(f"  {name}: {info['shape']} {info['dtype']}")

def main():
    """
    Main function
    """
    try:
        # Check if only listing datasets
        if os.getenv('LIST_DATASETS'):
            input_file = os.getenv('CATALOG_INPUT', '')
            if not input_file:
                raise ValueError("To list datasets, set CATALOG_INPUT environment variable")
            list_datasets(input_file)
            return
        
        # Get configuration parameters
        config = get_environment_variables()
        
        input_file = config['input_file']
        output_file = config['output_file']
        n_samples = config['n_samples']
        random_seed = config['random_seed']
        
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        print("=== HDF5 Stellar Catalog Sampling Tool ===")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Sample size: {n_samples}")
        print(f"Random seed: {random_seed}")
        
        # Open HDF5 file
        with h5py.File(input_file, 'r') as h5_in:
            total_galaxies = get_total_galaxies(h5_in)
            print(f"Total galaxies: {total_galaxies}")
            
            # Perform random sampling
            print("Randomly sampling galaxy data...")
            sampled_data = random_sample_galaxies(h5_in, n_samples, random_seed)
            
            # Display sampling information
            print(f"Successfully sampled {len(sampled_data)} datasets")
            for name, data in sampled_data.items():
                print(f"  {name}: {data.shape} {data.dtype}")
        
        # Save results
        save_sampled_data(sampled_data, output_file)
        
        print("Operation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage instructions:")
        print("1. Set required environment variables:")
        print("   export CATALOG_INPUT=/path/to/input.h5")
        print("   export CATALOG_OUTPUT=/path/to/output.h5")
        print("   export N_SAMPLES=1000")
        print("2. Optional environment variables:")
        print("   export RANDOM_SEED=42  # Set random seed for reproducibility")
        print("   export LIST_DATASETS=1  # Only list datasets and exit")
        print("3. Run script: python sample_galaxies.py")
        sys.exit(1)

if __name__ == "__main__":
    main()