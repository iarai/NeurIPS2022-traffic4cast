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
"""Generate 15 minute speeds for roads using the Traffic4cast Movie Grid data.

This script derives the 15 minute median speeds for each road segment using the
aggregated Traffic4cast Movie Grid data as source.

Prior to starting you should have generated the free flow edges data from the traffic map
movie data using the script `dp05_free_flow.py`.

The script expects the subfolders `movie_15min/<city>` and `road_graph/<city>` in the
passed data_folder. For every input file `<date>_<city>_8ch_15min.h5` a speed classes
output file `speed_classes_<date>.parquet` will be written to `speed_classes/<city>`.

Usage:
  dp06_speed_classes.py [-h] -d DATA_FOLDER -c CITY [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolder road_graph/<city> with `road_graph_freeflow.parquet
  -c city_name, --city city_name
                        Name of the city to be processed
  -dffn, --disable_free_flow_normalization
                        Disable normalizing free flow speeds using the speed limit
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import re
import sys
import warnings
from ast import literal_eval
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import geopandas
import numpy as np
import pandas
from h5_helpers import load_h5_file


MOVIE_MAXSPEED_KPH = 120  # Capped maximum speed in the Traffic Map Moview


SegmentSpeedStats = namedtuple("SegmentSpeedStats", ["volumes", "speeds", "median_speeds", "median_speeds_kph", "mean_speeds", "mean_speeds_kph"])


def extract_segment_speeds(data_aggregated_day, intersecting_cells):
    vols = np.zeros(shape=(data_aggregated_day.shape[0],))
    speeds = np.zeros(shape=(len(intersecting_cells), data_aggregated_day.shape[0]))
    # Intersecting cell tuples contain (row,column,heading,overlap)
    for i, (r, c, h, _) in enumerate(intersecting_cells):
        r = int(r)
        c = int(c)
        h = int(h)
        vols += data_aggregated_day[:, r, c, 2 * h]
        # TODO: possible improvement to use speed weighting by overlap here.
        speeds[i] = data_aggregated_day[:, r, c, 2 * h + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_speeds = np.nanmedian(speeds, axis=0)
        mean_speeds = np.nanmean(speeds, axis=0)
    median_speeds_kph = median_speeds / 255 * MOVIE_MAXSPEED_KPH
    mean_speeds_kph = mean_speeds / 255 * MOVIE_MAXSPEED_KPH
    return SegmentSpeedStats(vols, speeds, median_speeds, median_speeds_kph, mean_speeds, mean_speeds_kph)


# ATTENTION: also exists in prepare_training_data_cc.py, make sure to keep in sync
def free_flow_speed_limit(free_flow_kph, speed_limit_kph, min_free_flow_kph=20, min_speed_limit_kph=5, max_reduce=0.6):
    if not free_flow_kph or np.isnan(free_flow_kph) or free_flow_kph < min_free_flow_kph:
        free_flow_kph = min_free_flow_kph
    if speed_limit_kph >= min_speed_limit_kph and free_flow_kph > speed_limit_kph:
        free_flow_kph = speed_limit_kph
    # Reduce free flow to max 60% but not below, e.g. 32->20, 50->30, 80->48, 110->66
    free_flow_kph = max(free_flow_kph, speed_limit_kph * max_reduce)
    return free_flow_kph


def extract_sc_features(edges_df, date, data_aggregated_day, normalize_free_flow=True, slot_offset=0):  # noqa: C901
    road_graph_size = len(edges_df)
    result_data = []
    slots = data_aggregated_day.shape[0]
    for i, u, v, intersecting_cells, free_flow_kph, speed_limit_kph, gkey in zip(
        range(road_graph_size), edges_df["u"], edges_df["v"], edges_df["intersecting_cells"], edges_df["free_flow_kph"], edges_df["speed_kph"], edges_df["gkey"]
    ):
        intersecting_cells = literal_eval(intersecting_cells)  # List is stored as string
        speed_stats = extract_segment_speeds(data_aggregated_day, intersecting_cells)
        assert speed_stats.volumes.shape == (slots,), speed_stats.volumes.shape
        assert speed_stats.speeds.shape == (len(speed_stats.speeds), slots), speed_stats.speeds.shape
        for t in range(slots):
            median_speed = speed_stats.median_speeds[t]
            if np.isnan(median_speed) or median_speed == 0 or median_speed == 255:
                continue
            median_speed_kph = speed_stats.median_speeds_kph[t]
            if np.isnan(median_speed_kph) or median_speed_kph <= 0 or median_speed_kph >= MOVIE_MAXSPEED_KPH:
                continue
            probe_volume = speed_stats.volumes[t]
            if np.isnan(probe_volume) or probe_volume <= 0:
                continue

            normalized_free_flow_kph = free_flow_kph
            if normalize_free_flow:
                normalized_free_flow_kph = free_flow_speed_limit(free_flow_kph, speed_limit_kph)
            congestion_factor = median_speed_kph / normalized_free_flow_kph
            # Low congestion factor (aka traffic jam) needs more volume to be trusted. Otherwise could also just be a
            # stopping car. Such false-positive low speeds are very common while false-positive high speeds are
            # very rare and could only be generated by e.g. a motorcycle slipping through a traffic-jam.
            # Hence, if the speed is below 40% of free-flow we need at least 5 probe points to be convinced.
            # If the speed is up to 80% of free-flow we already trust the signal after 3 probe points.
            # Speeds above 80% are close to free flow and already a single probe is trustworthy enough.
            if probe_volume < 5 and congestion_factor < 0.4:
                continue
            if probe_volume < 3 and congestion_factor < 0.8:
                continue
            if probe_volume < 1:
                continue

            # Volume class is 1 for volumes 1 and 2; 3 for volumes 3 and 4; 5 for 5 and above
            volume_class = int(np.floor((np.clip(probe_volume, 1, 5) + 1) / 2) * 2 - 1)

            tt = slot_offset + t
            d = {
                "u": u,
                "v": v,
                "gkey": gkey,
                "day": date,
                "t": tt,
                "volume_class": volume_class,
                "median_speed_kph": median_speed_kph,
                "free_flow_kph": free_flow_kph,
            }
            result_data.append(d)
        if (i + 1) % 5000 == 0 or i >= road_graph_size - 1:
            print(f"\rProgress {i + 1}/{road_graph_size} ", end="")
    return result_data


def process_speed_classes(data_folder: Path, city: str, resume=False, normalize_free_flow=True, gpkg_date=None):
    edges_df = geopandas.read_parquet(data_folder / "road_graph" / city / "road_graph_freeflow.parquet")
    print(f"Loaded {len(edges_df)} edges with free flow speed for {city}")

    sc_city_folder = data_folder / "speed_classes" / city
    sc_city_folder.mkdir(parents=True, exist_ok=True)

    input_files = sorted((data_folder / "movie_15min" / city).rglob("*_8ch_15min.h5"))
    print(f"Found {len(input_files)} movie files, e.g. {input_files[0]}")
    for input_fn in input_files:
        date = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", str(input_fn)).group(1)
        output_fn = sc_city_folder / f"speed_classes_{date}.parquet"
        if resume and output_fn.exists() and date != gpkg_date:
            print(f"{output_fn} exists already ... skipping")
            continue
        data_aggregated_day = load_h5_file(input_fn)
        data = extract_sc_features(edges_df, date, data_aggregated_day, normalize_free_flow=normalize_free_flow)
        sc_df = pandas.DataFrame.from_dict(data)
        sc_df.to_parquet(output_fn, compression="snappy")
        print(f"Saved {len(sc_df)} speed_classes to {output_fn}")
        if date == gpkg_date:
            date_y, date_m, date_day = date.split("-")
            sc_edge_df = edges_df[["gkey", "geometry"]].merge(sc_df, on=["gkey"], how="right")
            sc_edge_df["timestamp"] = [
                datetime(year=int(date_y), month=int(date_m), day=int(date_day), hour=t // 4, minute=t % 4 * 15) for t in sc_edge_df["t"]
            ]
            sc_gpkg_fn = sc_city_folder / f"speed_classes_{date}.gpkg"
            sc_edge_df.to_file(sc_gpkg_fn, driver="GPKG", layer="speed_classes")
            print(f"Saved {len(sc_edge_df)} speed_classes GPKG to {sc_gpkg_fn}")

    print("... finished creating speed classes.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates cell mappings for road segments.")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data folder structure",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--city",
        type=str,
        help="City to be processed",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-dffn", "--disable_free_flow_normalization", help="Disable normalizing free flow speeds using the speed limit", required=False, action="store_true"
    )
    parser.add_argument(
        "-gpkg",
        "--gpkg_date",
        type=str,
        help="Generate GPKG output for the given date",
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
        normalize_free_flow = not params["disable_free_flow_normalization"]
        gpkg_date = params["gpkg_date"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    process_speed_classes(data_folder, city=city, resume=resume, normalize_free_flow=normalize_free_flow, gpkg_date=gpkg_date)


if __name__ == "__main__":
    main(sys.argv[1:])
