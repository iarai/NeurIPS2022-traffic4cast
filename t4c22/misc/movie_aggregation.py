#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Aggregate Traffic4cast Movies to 15 minute bins.

This script aggregates 5 minute movie time bins into 15 minute bins.

Prior to starting you should have downloaded and extracted the city
ZIP files:
- LONDON_2022.zip from https://developer.here.com/sample-data
- MADRID_2022.zip from https://developer.here.com/sample-data
- MELBOURNE_2022.zip from https://developer.here.com/sample-data

The output will be written to a newly created subfolder `movie_15min`
inside the input folder.

Usage:
  movie_aggregation.py [-h] -d MOVIE_CITY_FOLDER [-r]

Arguments:
  -d MOVIE_CITY_FOLDER, --data_folder MOVIE_CITY_FOLDER
                        Folder containing T4c movie data for a single city
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Union

import h5py
import numpy as np


NUM_SLOTS_NON_AGGREGATED = 288
NUM_SLOTS_AGGREGATED = 96
NUM_ROWS = 495
NUM_COLUMNS = 436


def load_h5_file(file_path: Union[str, Path]) -> np.ndarray:
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        return np.array(data)


def write_data_to_h5(data: np.ndarray, filename: Union[str, Path], compression="gzip", compression_level=9, dtype="uint8", verbose=False):
    with h5py.File(filename if isinstance(filename, str) else str(filename), "w", libver="latest") as f:
        if data.dtype != dtype:
            logging.warning(f"Found data with {data.dtype}, expected {dtype}.")
        if verbose:
            print(f"writing {filename} ...")
        f.create_dataset(
            # `chunks=(1, *data.shape[1:])`: optimize for row access!
            "array",
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
        )
        if verbose:
            print(f"... done writing {filename}")


def generate_15min_aggregates(data_folder: Path, resume=False):
    input_files = sorted((data_folder / "movie").rglob("*_8ch.h5"))
    print(f"{data_folder}: found {len(input_files)} movie files, e.g. {input_files[0]}")
    for input_fn in input_files:
        output_name = input_fn.name.replace("8ch", "8ch_15min")

        output_fn = data_folder / "movie_15min" / input_fn.relative_to(data_folder / "movie").parent / output_name
        output_fn.parent.mkdir(exist_ok=True, parents=True)
        print(f"Aggregating {output_fn} ...")
        if resume and output_fn.exists():
            print(f"{output_name} exist already ... skipping")
            continue

        data = load_h5_file(input_fn)
        data = data.astype(float)
        for k in range(4):
            data[..., k * 2 + 1] = np.where(data[..., k * 2] > 0, data[..., k * 2 + 1], np.nan)

        naancount = np.count_nonzero(np.isnan(data[..., [1, 3, 5, 7]]))
        assert naancount > 0, naancount

        assert naancount < NUM_SLOTS_NON_AGGREGATED * NUM_ROWS * NUM_COLUMNS * 4 - 9999, (naancount, NUM_SLOTS_NON_AGGREGATED * NUM_ROWS * NUM_COLUMNS * 4)
        assert np.count_nonzero(np.isnan(data[..., [0, 2, 4, 6]])) == 0

        data_resized = data.reshape(NUM_SLOTS_AGGREGATED, 3, NUM_ROWS, NUM_COLUMNS, 8)
        data_aggregated = np.zeros(shape=(NUM_SLOTS_AGGREGATED, NUM_ROWS, NUM_COLUMNS, 8))

        with warnings.catch_warnings():  # there can be empty slices, ignoring them e.g. at night
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data_aggregated[..., [0, 2, 4, 6]] = np.nansum(data_resized[..., [0, 2, 4, 6]], axis=1)
            data_aggregated[..., [1, 3, 5, 7]] = np.nanmean(data_resized[..., [1, 3, 5, 7]], axis=1)

        assert data_aggregated.shape == (NUM_SLOTS_AGGREGATED, NUM_ROWS, NUM_COLUMNS, 8), (data_aggregated.shape, input_fn)
        assert data_aggregated.dtype == np.float64, data_aggregated.dtype
        write_data_to_h5(data=data_aggregated, filename=output_fn, dtype=np.float64)
    print("... finished aggregating.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("This script takes a folder with 5 minute movie files and aggregates them to 15 minutes."))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c movie data",
        required=True,
    )
    parser.add_argument("-r", "--resume", help="Resume processing without regenerating existing files", required=False, action="store_true")
    return parser


def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(argv)
        params = vars(params)
        data_folder = Path(params["data_folder"])
        resume = params["resume"]
        generate_15min_aggregates(data_folder, resume)
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
