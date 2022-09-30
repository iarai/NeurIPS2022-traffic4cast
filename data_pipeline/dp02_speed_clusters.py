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
"""Generate Speed Clusters from Traffic4cast Movies.

This script generates Speed Clusters with the 5 most dominant speeds per
cell and heading.

Prior to starting you should have generated the 15 minute movie aggregates
using the script `dp01_movie_aggregation.py`.

The script expects a subfolder `movie_15min/<city>` in the passed data_folder.

The output will be written to a newly created subfolder `movie_speed_clusters`
inside the input folder.

Usage:
  dp02_speed_clusters.py [-h] -d DATA_FOLDER [-c CITY] [-n 20] [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolder movie_15min/<city> with
                        aggregated T4c movie data for at least one city
  -c city_name, --city city_name
                        Optional name of the city to be processed, otherwise
                        all city subfolders in movie_15min/ will be processed
  -n num_inputs, --num_inputs num_inputs
                        Optional number of aggregated movie data files to be
                        used for generating the clusters (default=20)
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path
from random import sample

import numpy as np
from ckmeans import ckmeans
from h5_helpers import load_h5_file
from h5_helpers import write_data_to_h5


NUM_SLOTS = 96
NUM_ROWS = 495
NUM_COLUMNS = 436


def create_clusters(data_folder: Path, city: str, num_inputs=20, min_num_speeds=10):
    files_15min = list((data_folder / "movie_15min" / city).glob("*_8ch_15min.h5"))
    if not files_15min:
        print(f'Found no input files in {(data_folder / "movie_15min" / city)}')
        return None
    files_15min = sample(files_15min, min(num_inputs, len(files_15min)))
    data_days = np.full((len(files_15min) * NUM_SLOTS, NUM_ROWS, NUM_COLUMNS, 8), np.nan, dtype=float)
    for i, file_name in enumerate(files_15min):
        data = load_h5_file(file_name)
        date = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", str(file_name)).group(1)
        print(f"{date} values > 0: {(data > 0).sum()}")
        data_days[i * NUM_SLOTS : i * NUM_SLOTS + NUM_SLOTS] = data
    print(f"Total values > 0: {(data_days > 0).sum()}")
    progress = 0
    speed_clusters = np.zeros(shape=(NUM_ROWS, NUM_COLUMNS, 4, 5, 2))
    for s, si in zip([1, 3, 5, 7], [0, 1, 2, 3]):
        for y in range(NUM_ROWS):
            for x in range(NUM_COLUMNS):
                progress += 1
                if progress % 10000 == 0:
                    print(f"\rProgress {progress}/{NUM_ROWS*NUM_COLUMNS*4}", end="")
                speeds = data_days[:, y, x, s]
                speeds = speeds[~np.isnan(speeds)]
                speeds = speeds.astype("float64")
                speeds = speeds / 255 * 120
                speeds = np.sort(speeds)
                if len(speeds) < min_num_speeds:
                    continue
                cluts = ckmeans(speeds, 5)
                ccents = [[np.median(c), len(c)] for c in cluts]
                speed_clusters[y][x][si] = ccents
    output_fn = data_folder / "movie_speed_clusters" / city / "speed_clusters.h5"
    output_fn.parent.mkdir(exist_ok=True, parents=True)
    write_data_to_h5(speed_clusters, output_fn, dtype=np.float64)
    return speed_clusters


def generate_speed_clusters(data_folder: Path, city: str, num_inputs=20, resume=False):
    if resume and (data_folder / "movie_speed_clusters" / city / "speed_clusters.h5").exists():
        print(f"Speed clusters file for {city} exists already. Skipping ...")
        return
    create_clusters(data_folder, city, num_inputs)
    print("... finished creating speed clusters.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates speed clusters from a folder with 15 minute movie files.")
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
    parser.add_argument(
        "-n",
        "--num_inputs",
        type=int,
        help="Number of input T4c movies to create clusters from",
        required=False,
        default=20,
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
        num_inputs = params["num_inputs"]
        resume = params["resume"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    generate_speed_clusters(data_folder, city=city, num_inputs=num_inputs, resume=resume)


if __name__ == "__main__":
    main(sys.argv[1:])
