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
"""Prepare Traffic4cast 2022 Training SuperSegment ETA Labels.

This script prepares supersegment eta training labels using speed_classes.

Prior to starting you should have downloaded and extracted the following 4
ZIP files to the same working data directory (<working_dir>):
- LONDON_2022.zip from https://developer.here.com/sample-data
- MADRID_2022.zip from https://developer.here.com/sample-data
- MELBOURNE_2022.zip from https://developer.here.com/sample-data
- T4C_INPUTS_2022.zip from http://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip
- T4C_INPUTS_ETA_2022.zip from http://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip

The script needs the following three folders in <data_folder>
├── road_graph
│   ├── london
│   ├── madrid
│   └── melbourne
├── speed_classes
│   ├── london
│   ├── madrid
│   └── melbourne
└── train
    ├── london
    ├── madrid
    └── melbourne

Usage:
  prepare_training_data_eta.py [-h] -d DATA_FOLDER [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing T4c data
  -r, --resume          Resume processing without regenerating existing files
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean

import pandas as pd


def compute_eta(edges, day_edge_speeds, t, usage_counts=None, debug=False):  # noqa:C901
    path_eta_s = 0
    cnt_current = 0
    cnt_free_flow = 0
    cnt_maxspeed = 0
    cnt_speedtoolow = 0
    debug_edges = []
    for ed in edges:
        e = ed["edge"]
        esp = day_edge_speeds[e]
        edge_speed_kph = esp["speeds"][t]
        speed_used = esp["sources"][t]
        if speed_used == "current":
            cnt_current += 1
        elif speed_used == "free_flow":
            cnt_free_flow += 1
        else:
            cnt_maxspeed += 1

        length_m = ed["length"]
        if edge_speed_kph < 0.5:
            cnt_speedtoolow += 1
            edge_speed_kph = 0.5
        edge_speed_mps = edge_speed_kph / 3.6
        edge_eta_s = length_m / edge_speed_mps

        if edge_eta_s > 1800:
            # If ETA longer than the 15 min slot and half of both neighbor time slots,
            # use 30 minutes plus the mean speed of all three slots.
            speeds = esp["speeds"][max(t - 1, 0) : t + 2]
            sources = esp["sources"][max(t - 1, 0) : t + 2]
            if debug:
                print(f"Debug {e}:")
                print(f"Bad eta {edge_eta_s} ({edge_speed_kph} kph)")
                print(f"Using neighbor speeds {speeds} (t={t}, {sources})")
            edge_speed_kph = mean(speeds)
            edge_speed_mps = edge_speed_kph / 3.6
            # New eta is 30 minutes plus the travel time as if neighboring speeds were used.
            # Capping at 45 minutes as this otherwise exceeds our prediction horizon.
            edge_eta_s = min(2400, 1800 + (length_m / edge_speed_mps))
            if debug:
                print(f"New eta {edge_eta_s} ({edge_speed_kph} kph)")

        path_eta_s += edge_eta_s
        if debug:
            debug_edges.append({"edge": e, "length_m": length_m, "edge_eta_s": edge_eta_s, "edge_speed_kph": edge_speed_kph, "speed_used": speed_used})

    if usage_counts:
        usage_counts["cnt_current"] += cnt_current
        usage_counts["cnt_free_flow"] += cnt_free_flow
        usage_counts["cnt_maxspeed"] += cnt_maxspeed
        usage_counts["cnt_speedtoolow"] += cnt_speedtoolow

    # Hard clip of path eta to 1h.
    path_eta_s = min(3600, path_eta_s)

    if debug:
        return {"eta_sec": path_eta_s, "edges": debug_edges}
    return {"eta_sec": path_eta_s}


def compute_eta_for_day(day, sc_parquet, supersegments, edge_maxspeeds_kph, edge_free_flows_kph, debug):
    sc_df = pd.read_parquet(sc_parquet)
    print(f"Read {len(sc_df)} rows from {sc_parquet}")

    maxspeed_cnt = 0
    edge_speeds = {}
    for uv, maxspeed in edge_maxspeeds_kph.items():
        if uv in edge_free_flows_kph:
            free_flow = edge_free_flows_kph[uv]
            speeds = [free_flow for _ in range(96)]
            sources = ["free_flow" for _ in range(96)]
        else:
            speeds = [maxspeed for _ in range(96)]
            sources = ["maxspeed" for _ in range(96)]
            maxspeed_cnt += 1
        edge_speeds[uv] = {"speeds": speeds, "sources": sources}
    print(f"{maxspeed_cnt} / {len(edge_speeds)} edges only have maxspeed")
    for t in range(0, 96):
        tsc_df = sc_df[sc_df["t"] == t]
        for u, v, ms in zip(tsc_df["u"], tsc_df["v"], tsc_df["median_speed_kph"]):
            esp = edge_speeds[(u, v)]
            esp["speeds"][t] = ms
            esp["sources"][t] = "current"

    print(" Start generating ETAs ...")
    eta_comp_time = 0
    usage_counts = {"cnt_current": 0, "cnt_free_flow": 0, "cnt_maxspeed": 0, "cnt_speedtoolow": 0}
    eta_results = []
    for t in range(0, 96):
        for s in supersegments:
            tstart = time.process_time()
            eta_info = compute_eta(s["edges"], edge_speeds, t, usage_counts=usage_counts, debug=debug)
            eta_comp_time += time.process_time() - tstart
            eta_result = {
                "identifier": s["identifier"],
                "day": day,
                "t": t,
                "eta": eta_info["eta_sec"],
            }
            if debug:
                eta_result["edges"] = eta_info["edges"]
            eta_results.append(eta_result)
    print(f" ... ETA comp took {eta_comp_time}s (on avg {eta_comp_time / len(eta_results)}s per call)")
    print(f"SuperSegments usage: {usage_counts}")

    eta_df = pd.DataFrame(eta_results)
    assert eta_df["eta"].max() <= 3600.0
    return eta_df


def generate_eta_labels(city, in_folder, out_folder, road_graph_folder: Path, resume, debug_list=None):  # noqa:C901
    rgss_file = road_graph_folder / city / "road_graph_supersegments.parquet"
    if not os.path.exists(rgss_file):
        print("ERROR: missing file {rgss_file}")
        return
    rgss_df = pd.read_parquet(rgss_file)
    print(f"Read {len(rgss_df)} supersegments")

    edges_file = road_graph_folder / city / "road_graph_edges.parquet"
    edges_df = pd.read_parquet(edges_file)
    print(f"Read {len(edges_df)} edges")

    edge_maxspeeds_kph = {}
    edge_lengths_m = {}
    speed_limit_field = "speed_kph"
    if city == "madrid":
        # The OSM speed_kph field in Madrid has parsing errors, let's use our own parsed version
        speed_limit_field = "parsed_maxspeed"
    for u, v, sl, lm in zip(edges_df["u"], edges_df["v"], edges_df[speed_limit_field], edges_df["length_meters"]):
        edge_maxspeeds_kph[(u, v)] = sl
        edge_lengths_m[(u, v)] = lm
    print(f"Read {len(edge_maxspeeds_kph)} edge max speed and length values")

    print("Extracting segment free flow speeds ...")
    free_flow_low = set()
    edge_free_flows_kph = {}
    sc_files = sorted((in_folder / city).glob("speed_classes_*.parquet"))
    for i in range(0, len(sc_files), 5):  # reading from every 5th file seems sufficient
        sc_df = pd.read_parquet(sc_files[i])
        for u, v, ff in zip(sc_df["u"], sc_df["v"], sc_df["free_flow_kph"]):
            if ff < 8 or ff != ff:  # Check for too low or NaN values
                free_flow_low.add((u, v))
                continue
            edge_free_flows_kph[(u, v)] = ff
    print(f"Free flow available for {len(edge_free_flows_kph)} segments ({len(free_flow_low)} values too low)")

    supersegments = []
    for identifier, nodes in zip(rgss_df["identifier"], rgss_df["nodes"]):
        edges = []
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            e = (n1, n2)
            edges.append(
                {
                    "edge": e,
                    "length": edge_lengths_m[e],
                }
            )
        supersegments.append({"identifier": identifier, "edges": edges})

    out_folder_city_labels = out_folder / city / "labels"
    print(f"Provisioning training output data to {out_folder_city_labels}")

    existing_dates = []
    if resume:
        existing_dates = [str(f).split("_")[-1][:-8] for f in out_folder_city_labels.glob(f"eta_labels_*.parquet")]

    eta_count = 0
    for i, sc_parquet in enumerate(sc_files):
        day = str(sc_parquet).split("_")[-1][:-8]
        if resume and day in existing_dates:
            print(f"eta_labels_{day}.parquet exists already ... skipping")
            continue
        debug = False
        if debug_list:
            debug = True
            if day not in debug_list:
                continue

        print(f"Processing labels file {i}/{len(sc_files)}")

        eta_df = compute_eta_for_day(day, sc_parquet, supersegments, edge_maxspeeds_kph, edge_free_flows_kph, debug)
        eta_count += len(eta_df)

        output_parquet = out_folder_city_labels / f"eta_labels_{day}.parquet"
        eta_df.to_parquet(output_parquet, compression="snappy")
        print(f"Wrote labels to {output_parquet}")

    if eta_count > 0:
        print(f"\nDone generating training output data to {out_folder_city_labels}")
        print(f" Schema: {list(eta_df.columns)}")
        print(f" Total rows: {eta_count}\n\n")


def generate_training_labels(data_folder: Path, resume):
    print(f"Processing label data in {data_folder}")
    for city in ["london", "madrid", "melbourne"]:
        data_folder_train_city_labels = data_folder / "train" / city / "labels"
        data_folder_train_city_labels.mkdir(exist_ok=True, parents=True)
        generate_eta_labels(
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
