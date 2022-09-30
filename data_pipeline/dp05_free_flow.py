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
"""Generate free flow speeds for roads using the Traffic4cast Movie Grid data.

This script derives free flow speeds for each road segment using the speeds clusters
in the Traffic4cast Movie Grid data.

Prior to starting you should have generated the speed clusters from the traffic map
movie data using the script `dp02_speed_clusters.py`.

Prior to starting you also should have generated the road graph edge and cell mappings
data using the scripts `dp03_road_graph.py` and `dp04_intersecing_cells.py` or downloaded
and extracted the existing files from the Traffic4cast competition ZIP files:
- T4C_INPUTS_2022.zip from http://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip
- T4C_INPUTS_ETA_2022.zip from http://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip

The script expects a subfolder `road_graph/<city>` in the passed data_folder.

The output file `road_graph_freeflow.parquet` will be written to `road_graph/<city>`.

Usage:
  dp05_free_flow.py [-h] -d DATA_FOLDER -c CITY [-sl] [-f]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolder road_graph/<city> with
                        road_graph_edges.parquet and road_graph_geometries.parquet
  -c city_name, --city city_name
                        Name of the city to be processed
  -sl, --use_speed_limit
                        Use the official speed limit to cap the free flow speed
  -f, --force_overwrite
                        Force re-processing and overwriting existing free flow files
"""
import argparse
import logging
import os
import sys
import warnings
from ast import literal_eval
from pathlib import Path

import geopandas
import numpy as np
from data_helpers import invert_heading
from h5_helpers import load_h5_file
from shapely import wkb


def extract_merged_speed_clusters(speed_clusters, intersecting_cells):
    merged_speed_clusters = []
    for r, c, h, _ in intersecting_cells:
        r = int(r)
        c = int(c)
        h = int(h)
        merged_speed_clusters.extend([list(x) for x in speed_clusters[r, c, h, :]])
    merged_speed_clusters = sorted(merged_speed_clusters)
    return np.array(merged_speed_clusters)


def get_free_flow(merged_speed_clusters, speed_limit=-1, volume_threshold=0.2):
    lhs = len(merged_speed_clusters) - 1
    if merged_speed_clusters[lhs, 0] <= speed_limit:
        return merged_speed_clusters[lhs, 0]
    num_speeds = sum(merged_speed_clusters[:, 1])
    if num_speeds == 0:
        return np.nan
    vol_num_speeds = 0
    for i in range(lhs, -1, -1):
        vol_num_speeds += merged_speed_clusters[i, 1]
        if vol_num_speeds / num_speeds > volume_threshold:
            free_flow = merged_speed_clusters[i, 0]
            return free_flow
    return np.nan


def merge_speed_clusters_from_intersecting_cells(speed_clusters, intersecting_cells, oneway, speed_limit_kph=-1, debug=False):
    intersecting_cells = literal_eval(intersecting_cells)
    if len(intersecting_cells) == 0:
        return -1
    # For two-way also add the opposite headings.
    all_intersecting_cells = intersecting_cells[:]
    if not oneway or oneway == "False":
        for ic in intersecting_cells:
            all_intersecting_cells.append((ic[0], ic[1], invert_heading(int(ic[2])), ic[3]))
    # Aggregate the speed clusters.
    merged_speed_clusters = extract_merged_speed_clusters(speed_clusters, all_intersecting_cells)
    free_flow = get_free_flow(merged_speed_clusters, speed_limit_kph)
    if debug:
        print(f"speed_limit_kph: {speed_limit_kph}")
        print(f"oneway: {oneway} (type={type(oneway)})")
        print(f"all_intersecting_cells (len={len(all_intersecting_cells)}): {all_intersecting_cells}")
        print(f"merged_speed_clusters: {merged_speed_clusters}")
        print(f"free_flow: {free_flow}")
    return free_flow


def generate_free_flow_edges(speed_clusters, df, city, use_speed_limit=True, debug=False):
    if use_speed_limit:
        df["free_flow_kph"] = [
            merge_speed_clusters_from_intersecting_cells(speed_clusters, ic, oneway, speed_limit_kph, debug)
            for ic, oneway, speed_limit_kph in zip(df["intersecting_cells"], df["oneway"], df["speed_kph"])
        ]
    else:
        df["free_flow_kph"] = [
            merge_speed_clusters_from_intersecting_cells(speed_clusters, ic, oneway, debug=debug) for ic, oneway in zip(df["intersecting_cells"], df["oneway"])
        ]
    print(f'{len(df[df["free_flow_kph"].isna()])}/{len(df)} edges in {city} have NaN freeflow')
    return df


def generate_free_flow(data_folder: Path, city: str, use_speed_limit=False, generate_gpkg=True):
    speed_clusters = load_h5_file(data_folder / "movie_speed_clusters" / city / "speed_clusters.h5")
    print(f"Loaded speed clusters {speed_clusters.shape} for {city}")
    df = geopandas.read_parquet(data_folder / "road_graph" / city / "road_graph_intersecting_cells.parquet")
    print(f"Loaded {len(df)} edges with intersecting cells for {city}")
    df["gkey"] = [hash(wkb.dumps(g)) for g in df["geometry"]]
    df = generate_free_flow_edges(speed_clusters, df, city, use_speed_limit=use_speed_limit)
    df = df[
        [
            "u",
            "v",
            "gkey",
            "osmid",
            "speed_kph",
            "maxspeed",
            "highway",
            "oneway",
            "lanes",
            "tunnel",
            "length_meters",
            "geometry",
            "intersecting_cells",
            "free_flow_kph",
        ]
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        rg_fn = data_folder / "road_graph" / city / "road_graph_freeflow.parquet"
        df.to_parquet(rg_fn, compression="snappy")
        print(f"Saved freeflow to {rg_fn}")
    if generate_gpkg:
        rg_gpkg_fn = data_folder / "road_graph" / city / "road_graph_freeflow.gpkg"
        df.to_file(rg_gpkg_fn, driver="GPKG", layer="edges_ff")
        print(f"Saved freeflow GPKG to {rg_gpkg_fn}")
    return df


def process_free_flow(data_folder: Path, city: str, use_speed_limit=False, overwrite=False):
    if not overwrite and (data_folder / "road_graph" / city / "road_graph_freeflow.parquet").exists():
        print(f"Free flow file for {city} exists already. Skipping ...")
        return
    generate_free_flow(data_folder, city, use_speed_limit)
    print("... finished creating free flow speeds.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates free flow speeds for road segments.")
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
    parser.add_argument("-sl", "--use_speed_limit", help="Use speed limit for capping", required=False, action="store_true")
    parser.add_argument("-f", "--force_overwrite", help="Force overwriting existing files", required=False, action="store_true")
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
        use_speed_limit = params["use_speed_limit"]
        overwrite = params["force_overwrite"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    process_free_flow(data_folder, city=city, use_speed_limit=use_speed_limit, overwrite=overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
