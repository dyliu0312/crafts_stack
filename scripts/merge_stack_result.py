import os
from itertools import product

from mytools.data import get_filename, get_stacked_result, h5


def get_filepath_format(base, pattern, *args):
    """
    Generate file paths by combining base directory, pattern string, and arguments.

    This function processes positional arguments that may contain space-separated values
    and generates all possible combinations of file paths based on the provided pattern.

    Args:
        base (str): Base directory path
        pattern (str): Format pattern string containing {} placeholders
        *args: Variable-length argument list. String arguments containing spaces
               will be split into multiple values.

    Returns:
        list: A list of generated full file paths

    Example:
        >>> base = '/home/user/data'
        >>> pattern = "{}_df{}_rmfg{}_xsize{}_nfs{}.h5"
        >>> get_filepath_format(base, pattern, 'tng', '120k', '0 1', '3000', '10')
        ['/home/user/data/tng_df120k_rmfg0_xsize3000_nfs10.h5',
         '/home/user/data/tng_df120k_rmfg1_xsize3000_nfs10.h5']

    Note:
        - The number of {} placeholders in pattern must match the number of arguments
        - Space-separated values in arguments will be split and treated as multiple values
        - All possible combinations of the split values will be generated
    """
    # Process arguments: split strings with spaces into lists, convert others to lists
    processed_args = []
    for arg in args:
        if isinstance(arg, str) and " " in arg:
            processed_args.append(arg.split())
        else:
            processed_args.append([str(arg)])

    # Generate all possible combinations of arguments
    filepaths = []
    for combination in product(*processed_args):
        filepath = os.path.join(base, pattern.format(*combination))
        filepaths.append(filepath)

    return filepaths


def merge_stack_result(
    outpath, resultpath, dest_keys=["Signal", "Mask"]
):
    """
    Merge the stacked result from different models into one file.
    """
    if not outpath.endswith(".h5"):
        raise ValueError("Output file must be a h5 file")

    with h5.File(outpath, "a") as f:
        for path in resultpath:
            if not os.path.exists(path):
                print(f"[X] File {path} does not exist")
                continue
            if not path.endswith(".h5"):
                print(f"[X] {path} is not a h5 file")
                continue
            try:
                result = get_stacked_result(path, *dest_keys)
            except Exception as e:
                print(f"[X] Error occurred while getting stacked result from {path}: {e}")
                continue
            fname = get_filename(path)
            if fname in f:
                print(f"[X] File {fname} already exists in {outpath}")
                continue
            else:
                grp = f.create_group(fname)
                grp.create_dataset(dest_keys[0], data=result.data, compression="gzip")
                grp.create_dataset(dest_keys[1], data=result.mask) # type: ignore
                print(f"✅ Merged result {fname} to {outpath}")
    return None


def main():
    """
    Main function to get the stacked result.
    """
    base = os.getenv("BASE", "./")
    fname_pattern = os.getenv("FNAME_PATTERN", "{}_df{}_rmfg{}_xsize{}_nfs{}.h5")
    args = os.getenv("ARGS", "tng,120k,0,3000,10").split(",")
    outpath = os.getenv("OUTPATH", "./merged_stack_result.h5")
    dest_keys = os.getenv("DEST_KEYS", "Signal,Mask").split(",")

    resultpath = get_filepath_format(base, fname_pattern, *args)
    merge_stack_result(outpath, resultpath, dest_keys)
    return None


if __name__ == "__main__":
    main()
