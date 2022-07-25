# -*- coding: utf-8 -*-
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
"""Prepare Traffic4cast 2022 Training Labels.

This script prepares congestion class training labels using speed_classes.

Prior to starting you should have downloaded and extracted the following 4
ZIP files to the same working data directory (<working_dir>):
- LONDON_2022.zip from https://developer.here.com/sample-data
- MADRID_2022.zip from https://developer.here.com/sample-data
- MELBOURNE_2022.zip from https://developer.here.com/sample-data
- T4C_INPUTS_2022.zip from http://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip

The script needs the following two folders in <data_folder>
├── speed_classes
│   ├── london
│   ├── madrid
│   └── melbourne
└── train
    ├── london
    ├── madrid
    └── melbourne

Usage:
  prepare_training_data_cc.py [-h] -d DATA_FOLDER [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing T4c data
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -


# ATTENTION: copied from t4c22_03_freeflow_speeds_cls.py, make sure to keep in sync
def free_flow_speed_limit(free_flow_kph, speed_limit_kph):
    if not free_flow_kph or np.isnan(free_flow_kph) or free_flow_kph < 20:
        free_flow_kph = 20
    if speed_limit_kph >= 5 and free_flow_kph > speed_limit_kph:
        free_flow_kph = speed_limit_kph
    # Reduce free flow to max 60% but not below, e.g. 32->20, 50->30, 80->48, 110->66
    free_flow_kph = max(free_flow_kph, speed_limit_kph * 0.6)
    return free_flow_kph


# ATTENTION: copied from t4c22_04_congestion_classes_cls.py, make sure to keep in sync
def compute_cc_defensive(median_speed_kph, freeflow_speed_kph, probe_volume):
    if median_speed_kph <= 0:
        return 0
    if median_speed_kph >= 120:
        return 0
    assert freeflow_speed_kph > 0, (median_speed_kph, freeflow_speed_kph, probe_volume)
    congestion_factor = median_speed_kph / freeflow_speed_kph
    if congestion_factor < 0.4 and probe_volume >= 5:
        return 3
    elif congestion_factor >= 0.4 and congestion_factor < 0.8 and probe_volume >= 3:
        return 2
    elif congestion_factor >= 0.8 and probe_volume > 0:
        return 1
    else:
        return 0


def compute_cc(median_speed_kph, free_flow_kph, speed_limit_kph, volume_class):
    free_flow_kph = free_flow_speed_limit(free_flow_kph, speed_limit_kph)
    return compute_cc_defensive(median_speed_kph, free_flow_kph, volume_class)


def generate_cc_labels(city, in_folder, out_folder, road_graph_folder: Path, resume):
    out_folder_city_labels = out_folder / city / "labels"
    print(f"Provisioning training output data to {out_folder_city_labels}")
    edges_file = road_graph_folder / city / "road_graph_edges.parquet"
    edges_df = pd.read_parquet(edges_file)
    print(f"Read {len(edges_df)} edges")
    cc_count = 0
    sc_files = sorted((in_folder / city).glob("speed_classes_*.parquet"))
    existing_dates = []
    if resume:
        existing_dates = [f.split("_")[-1][:-8] for f in out_folder_city_labels.glob(f"cc_labels_*.parquet")]
    for i, sc_parquet in enumerate(sc_files):
        day = str(sc_parquet).split("_")[-1][:-8]
        if resume and day in existing_dates:
            print(f"cc_labels_{day}.parquet exist already ... skipping")
            continue
        print(f"Processing labels file {i}/{len(sc_files)}")
        sc_df = pd.read_parquet(sc_parquet)
        print(f"Read {len(sc_df)} rows from {sc_parquet}")
        cc_df = sc_df.merge(edges_df, on=["u", "v"])
        speed_limit_field = "speed_kph"
        if city == "madrid":
            # The OSM speed_kph field in Madrid has parsing errors, let's use our own parsed version
            speed_limit_field = "parsed_maxspeed"
        cc_df["cc"] = [
            compute_cc(ms, ff, sl, vc)
            for ms, ff, sl, vc in zip(cc_df["median_speed_kph"], cc_df["free_flow_kph"], cc_df[speed_limit_field], cc_df["volume_class"])
        ]
        cc_df = cc_df.reset_index()
        cc_df = cc_df[["u", "v", "day", "t", "cc"]]
        cc_df = cc_df[cc_df["cc"] > 0]
        cc_count += len(cc_df)
        print(f' Congestion classes: {dict(cc_df[["cc"]].groupby(["cc"]).size())}')
        print(f' Segments: {len(cc_df.groupby(["u","v"]).count())}')
        output_parquet = out_folder_city_labels / f"cc_labels_{day}.parquet"
        cc_df.to_parquet(output_parquet, compression="snappy")
        print(f"Wrote labels to {output_parquet}")
    if cc_count > 0:
        print(f"\nDone generating training output data to {out_folder_city_labels}")
        print(f" Schema: {list(cc_df.columns)}")
        print(f" Total rows: {cc_count}\n\n")


def generate_training_labels(data_folder: Path, resume):
    print(f"Processing label data in {data_folder}")
    for city in ["london", "madrid", "melbourne"]:
        data_folder_train_city_labels = data_folder / "train" / city / "labels"
        data_folder_train_city_labels.mkdir(exist_ok=True, parents=True)
        generate_cc_labels(
            city,
            in_folder=data_folder / "speed_classes",
            out_folder=data_folder / "train",
            road_graph_folder=data_folder / "road_graph",
            resume=resume,
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("This script takes the T4c22 data directory and prepares the training label files."))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data",
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
        generate_training_labels(data_folder, resume)
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
