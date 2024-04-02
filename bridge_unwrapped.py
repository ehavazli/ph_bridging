#!/usr/bin/env python
import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio as rio
from dolphin._log import get_log
from src.conncomp import connectComponent

logger = get_log(__name__)


EXAMPLE = """Example usage:
    bridge_unwrapped.py input_folder output_folder -p 4
"""


def create_parser() -> argparse.ArgumentParser:
    """Create parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool for phase bridging",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE,
    )

    parser.add_argument("input_folder", type=Path, help="Path to the input folder")

    parser.add_argument("output_folder", type=Path, help="Path to the output folder")

    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel processes to use (default=4)",
    )

    return parser


def get_unw_conncomp(
    unw_file: str, conncomp_file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Get data from unwrapped and connected component files

    Args:
        unw_file (str): Path to the unwrapped file.
        conncomp_file (str): Path to the connected component file.

    Returns:
        Tuple containing:
        - unw_data (numpy.ndarray): Unwrapped phase.
        - conncomp_data (numpy.ndarray): Connected components.
        - amp_data (numpy.ndarray): Amplitude data, if present.
        - unw_profile (dict): GDAL profile of unwrapped file (to be used for new file).
        - transform (list): GDAL transform of unwrapped file.
    """
    with rio.open(unw_file) as unw:
        unw_profile = unw.profile
        unw_profile.transform = unw.transform
        if unw_profile["count"] == 1:
            amp_data = np.nan
            unw_data = unw.read(1)
        if unw_profile["count"] == 2:
            amp_data = unw.read(1)
            unw_data = unw.read(2)

    with rio.open(conncomp_file) as conncomp:
        conncomp_data = conncomp.read(1)

    return unw_data, conncomp_data, amp_data, unw_profile


def write_file(unw_arr, amp_arr, profile, new_filename):
    if np.isnan(amp_arr):
        del amp_arr

    with rio.open(new_filename, "w", **profile) as dst:
        if profile["count"] == 1:
            dst.write(unw_arr, 1)
        if profile["count"] == 2:
            dst.write(amp_arr, 1)
            dst.write(unw_arr, 2)
    logger.info("Wrote: %s", new_filename)


def bridge_iteration(unw_file: Path, conncomp_file: Path, output_folder: Path) -> None:
    """
    Perform bridging on unwrapped interferograms.

    Args:
        unw_file (Path): Path to the unwrapped file.
        conncomp_file (Path): Path to the connected component file.
        output_folder (Path): Path to the output folder.
    """
    logger.info("Processing unw file: %s, conncomp file: %s", unw_file, conncomp_file)
    unw, conncomp, amp, profile = get_unw_conncomp(str(unw_file), str(conncomp_file))
    cc = connectComponent(conncomp=conncomp, metadata=profile)
    brdg_labels = cc.label()
    bridges = cc.find_mst_bridge()
    bridge_unw = cc.unwrap_conn_comp(unw)

    outfile_name = output_folder / f"{unw_file.stem}_brdg_unw{unw_file.suffix}"
    write_file(bridge_unw, amp, profile, str(outfile_name))


def main():
    """Run the program."""
    parser = create_parser()
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # Validate input and output folders
    if not input_folder.is_dir():
        logger.error(
            "Input folder '%s' does not exist or is not a directory.", input_folder
        )
        return
    if not output_folder.is_dir():
        logger.info(
            "Output folder '%s' does not exist or is not a directory.", output_folder
        )
        logger.info("Creating output folder '%s'", output_folder)
        # Create output folder if it doesn't exist
        output_folder.mkdir(exist_ok=True)

    unw_file_list = sorted(input_folder.glob("*_unw.tif"))
    conncomp_file_list = sorted(input_folder.glob("*_unw_conncomp.tif"))

    st = time.time()
    with mp.Pool(processes=args.parallel) as pool:
        # Map the process_iteration function to each pair
        pool.starmap(
            bridge_iteration,
            [
                (unw, conncomp, output_folder)
                for unw, conncomp in zip(unw_file_list, conncomp_file_list)
            ],
        )
    et = time.time()
    elapsed_time = (et - st) / 60
    logger.info("Elapsed time: %.2f minutes", elapsed_time)


if __name__ == "__main__":
    main()
