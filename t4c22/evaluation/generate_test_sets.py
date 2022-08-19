#!/usr/bin/python3
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
#
#   http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def day_t_filter(day, t, day_whitelist=None, t_whitelist=None, weekday_whitelist=None):
    if day_whitelist is not None:
        return day in day_whitelist
    if t_whitelist is not None:
        return t in t_whitelist
    if weekday_whitelist is not None:
        weekday = datetime.datetime.strptime(day, "%Y-%m-%d").weekday()
        return weekday in weekday_whitelist
    return True


def day_t_filter_sample(day_t, low_incl=6 * 4, high_excl=22 * 4, size=10, weekday_whitelist=None):
    day_t_filtered = [(day, t) for (day, t) in day_t if day_t_filter(day, t, t_whitelist=set(range(low_incl, high_excl)), weekday_whitelist=weekday_whitelist)]
    indices = np.random.choice(len(day_t_filtered), size=size, replace=False)
    day_t_filtered_sampled = [day_t_filtered[i] for i in indices]
    return day_t_filtered_sampled


def generate_test_sets(city, DATA_DAYS, WITHHELD_GOLDEN_FOLDER, WITHHELD_TEST_FOLDER, TEST_FOLDER, blacklist=None):
    if blacklist is None:
        blacklist = []
    day_t = [(day, t) for day in DATA_DAYS[city]["test"] if day not in blacklist for t in range(4, 96)]
    day_t_filtered_sampled = day_t_filter_sample(
        day_t,
        low_incl=6 * 4,
        high_excl=22 * 4,
        weekday_whitelist=set(range(7)),
        size=100,
    )
    day_t_filtered_sampled = [(slot, day, t) for slot, (day, t) in enumerate(day_t_filtered_sampled)]
    with open(f"{WITHHELD_GOLDEN_FOLDER}/{city}/sampled_day_t.json", "w") as f:
        json.dump(day_t_filtered_sampled, f)
    print(f"Wrote test day and t mappings to {WITHHELD_GOLDEN_FOLDER}/{city}/sampled_day_t.json")

    bad_labels = set()
    input_dfs = []
    label_dfs = []
    for idx, day, t in day_t_filtered_sampled:
        input_df = pd.read_parquet(f"{WITHHELD_TEST_FOLDER}/{city}/input/counters_{day}.parquet")
        input_df = input_df[input_df["t"] == t]
        input_df["test_idx"] = idx
        input_df = input_df.drop(["day", "t"], axis=1)
        input_dfs.append(input_df)

        num_inputs = len(input_df)
        if num_inputs < 2000:
            print(f"Bad counter inputs {city} {idx} ({day}, {t}): {num_inputs}")

        label_df = pd.read_parquet(f"{WITHHELD_TEST_FOLDER}/{city}/labels/cc_labels_{day}.parquet")
        label_df = label_df[label_df["t"] == t]
        label_df["test_idx"] = idx
        label_df = label_df.drop(["day", "t"], axis=1)
        label_dfs.append(label_df)

        num_label_segments = len(label_df.groupby(["u", "v"]).count())
        if num_label_segments < 2000:
            print(f"Bad golden labels {city} {idx} ({day}, {t})")
            print(f' Congestion classes: {dict(label_df[["cc"]].groupby(["cc"]).size())}')
            print(f" Segments: {num_label_segments}")
            bad_labels.add(day)

    if len(bad_labels) > 0:
        print(sorted(bad_labels))

    input_df = pd.concat(input_dfs)
    print(f"Sampled {len(input_df)} inputs for {len(input_dfs)} slots")
    input_parquet = f"{TEST_FOLDER}/{city}/input/counters_test.parquet"
    input_df.to_parquet(input_parquet, compression="snappy")
    print(f"Wrote sampled input counters to {input_parquet}")

    label_df = pd.concat(label_dfs)
    print(f"Sampled {len(label_df)} labels for {len(label_dfs)} slots")
    label_parquet = f"{WITHHELD_GOLDEN_FOLDER}/{city}/labels/cc_labels_test.parquet"
    label_df.to_parquet(label_parquet, compression="snappy")
    print(f"Wrote sampled golden labels to {label_parquet}")


def get_chunked(arr, chunk_size=7):
    return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]


def every_other(arr, start=0):
    arr = get_chunked(arr)
    for i in range(start, len(arr), 2):
        yield from arr[i]


def unique_months(arr):
    months = set()
    for d in arr:
        months.add(d[:7])
    return sorted(months)


CITIES = ["london", "madrid", "melbourne"]


def create_city_folders(parent, subfolders=None):
    if subfolders is None:
        subfolders = []
    for city in CITIES:
        os.makedirs(f"{parent}/{city}", exist_ok=True)
        for sf in subfolders:
            os.makedirs(f"{parent}/{city}/{sf}", exist_ok=True)


LONDON_DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2019-07-01", "2020-01-31")]
MELBOURNE_DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-06-01", "2020-12-30")]
MADRID_DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2021-06-01", "2021-12-31")]
DATA_DAYS = {
    "madrid": {"months": unique_months(MADRID_DAYS), "train": list(every_other(MADRID_DAYS, 0)), "test": list(every_other(MADRID_DAYS, 1))},
    "melbourne": {"months": unique_months(MELBOURNE_DAYS), "train": list(every_other(MELBOURNE_DAYS, 0)), "test": list(every_other(MELBOURNE_DAYS, 1))},
    "london": {"months": unique_months(LONDON_DAYS), "train": list(every_other(LONDON_DAYS, 0)), "test": list(every_other(LONDON_DAYS, 1))},
}

BLACKLIST = [
    # London missing counters
    "2019-09-19",
    "2019-09-20",
    "2019-09-21",
    "2019-09-22",
    "2019-12-14",
    # London missing labels
    "2019-10-04",
    "2019-10-05",
    "2019-10-06",
    # Madrid missing labels
    "2021-06-25",
    "2021-06-27",
    "2021-08-04",
    # Melbourne missing counters
    "2020-08-07",
    "2020-10-04",
    # Melbourne missing labels
    "2020-07-11",
    "2020-07-12",
    "2020-07-23",
    "2020-08-05",
    "2020-08-06",
    "2020-08-08",
    "2020-08-09",
    "2020-08-31",
    "2020-09-05",
    "2020-09-06",
    "2020-09-18",
    "2020-09-20",
    "2020-10-18",
    "2020-09-01",
    "2020-09-14",
    "2020-10-03",
    "2020-10-26",
    "2020-11-01",
]


def prepare_and_generate_test_sets(release_folder: Path):  # noqa:C901

    # HERE Data
    movie_folder = release_folder / "movie"
    create_city_folders(movie_folder)

    speed_classes_folder = release_folder / "speed_classes"
    create_city_folders(speed_classes_folder)

    # Open Data
    road_graph_folder = release_folder / "road_graph"
    create_city_folders(road_graph_folder)

    loop_counter_folder = release_folder / "loop_counter"
    create_city_folders(loop_counter_folder)

    # Generated Data
    train_folder = release_folder / "train"
    create_city_folders(train_folder, ["input", "labels"])

    test_folder = release_folder / "test"
    create_city_folders(test_folder, ["input"])

    # Withheld Data
    withheld_golden_folder = release_folder / "withheld/golden"
    create_city_folders(withheld_golden_folder, ["labels"])

    withheld_sc_folder = release_folder / "withheld/speed_classes"
    create_city_folders(withheld_sc_folder)

    withheld_test_folder = release_folder / "withheld/test"
    create_city_folders(withheld_test_folder, ["input", "labels"])

    zip_folder = release_folder / "zip"
    zip_folder.mkdir(exist_ok=True, parents=True)

    # Empty
    submissions_folder = release_folder / "submissions"
    create_city_folders(submissions_folder)

    for city in CITIES:
        generate_test_sets(
            city,
            DATA_DAYS,
            withheld_golden_folder,
            withheld_test_folder,
            test_folder,
            blacklist=BLACKLIST,
        )


def create_parser() -> argparse.ArgumentParser:
    """Create test files and copy static and dynamic h5 files to the same place
    and tar them."""
    parser = argparse.ArgumentParser(description=("This scripts samples a test set from the withheld folder"))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data",
        required=True,
    )
    return parser


def main(args):  # noqa C901
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(args)
        params = vars(params)
        prepare_and_generate_test_sets(**params)

    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
