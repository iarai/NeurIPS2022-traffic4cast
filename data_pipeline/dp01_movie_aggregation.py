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

Prior to starting you should have downloaded and extracted the ZIP file
for the city of interest, e.g.:
- LONDON_2022.zip from https://developer.here.com/sample-data
- MADRID_2022.zip from https://developer.here.com/sample-data
- MELBOURNE_2022.zip from https://developer.here.com/sample-data

The script expects a subfolder `movie/<city>` in the passed data_folder.

The output will be written to a newly created subfolder `movie_15min`
inside the input folder.

Usage:
  dp01_movie_aggregation.py [-h] -d DATA_FOLDER [-c CITY] [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolder `movie/<city>` with
                        T4c movie data for at least one city
  -c city_name, --city city_name
                        Optional name of the city to be processed, otherwise
                        all city subfolders in movie/ will be processed
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from data_helpers import NUM_COLUMNS
from data_helpers import NUM_ROWS
from h5_helpers import load_h5_file
from h5_helpers import write_data_to_h5

NUM_SLOTS_NON_AGGREGATED = 288
NUM_SLOTS_AGGREGATED = 96


def generate_15min_aggregate(data: np.ndarray):
    data = data.astype(float)
    # backward compatibility for _9ch.h5 from 2020 competition data
    data = data[..., :8]
    for k in range(4):
        data[..., k * 2 + 1] = np.where(data[..., k * 2] > 0, data[..., k * 2 + 1], np.nan)

    nancount = np.count_nonzero(np.isnan(data[..., [1, 3, 5, 7]]))
    assert nancount > 0, nancount

    # We expect at least 10K valid values (hence expected number of values - 9999)
    assert nancount < NUM_SLOTS_NON_AGGREGATED * NUM_ROWS * NUM_COLUMNS * 4 - 9999, (nancount, NUM_SLOTS_NON_AGGREGATED * NUM_ROWS * NUM_COLUMNS * 4)
    assert np.count_nonzero(np.isnan(data[..., [0, 2, 4, 6]])) == 0

    data_resized = data.reshape(NUM_SLOTS_AGGREGATED, 3, NUM_ROWS, NUM_COLUMNS, 8)
    data_aggregated = np.zeros(shape=(NUM_SLOTS_AGGREGATED, NUM_ROWS, NUM_COLUMNS, 8))

    with warnings.catch_warnings():  # there can be empty slices, ignoring them e.g. at night
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data_aggregated[..., [0, 2, 4, 6]] = np.sum(data_resized[..., [0, 2, 4, 6]], axis=1)
        data_aggregated[..., [1, 3, 5, 7]] = np.nanmean(data_resized[..., [1, 3, 5, 7]], axis=1)

    assert data_aggregated.shape == (NUM_SLOTS_AGGREGATED, NUM_ROWS, NUM_COLUMNS, 8), data_aggregated.shape
    assert data_aggregated.dtype == np.float64, data_aggregated.dtype
    return data_aggregated


def generate_15min_aggregates(data_folder: Path, city=None, resume=False):
    if city:
        input_files = sorted((data_folder / "movie" / city).rglob("*_[8-9]ch.h5"))
    else:
        input_files = sorted((data_folder / "movie").rglob("*_[8-9]ch.h5"))
    print(f"{data_folder}: found {len(input_files)} movie files, e.g. {input_files[0]}")
    for input_fn in input_files:
        output_name = input_fn.name.replace("8ch", "8ch_15min").replace("9ch", "8ch_15min")

        output_fn = data_folder / "movie_15min" / input_fn.relative_to(data_folder / "movie").parent / output_name
        output_fn.parent.mkdir(exist_ok=True, parents=True)
        if resume and output_fn.exists():
            print(f"{output_name} exists already ... skipping")
            continue
        print(f"Aggregating {output_fn} ...")

        data = load_h5_file(input_fn)
        data_aggregated = generate_15min_aggregate(data)
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
    parser.add_argument(
        "-c",
        "--city",
        type=str,
        help="Limit processing to city",
        required=False,
        default=None,
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
        city = params["city"]
        resume = params["resume"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    generate_15min_aggregates(data_folder, city=city, resume=resume)


if __name__ == "__main__":
    main(sys.argv[1:])
