import os

from mytools.data import get_filename, get_stacked_result, h5

from crafts_stack.utils import get_filepath_format


def merge_stack_result(outpath, resultpath, dest_keys=["Signal", "Mask"]):
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
                print(
                    f"[X] Error occurred while getting stacked result from {path}: {e}"
                )
                continue
            fname = get_filename(path)
            if fname in f:
                print(f"[X] File {fname} already exists in {outpath}")
                continue
            else:
                grp = f.create_group(fname)
                grp.create_dataset(dest_keys[0], data=result.data, compression="gzip")
                grp.create_dataset(dest_keys[1], data=result.mask)  # type: ignore
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
