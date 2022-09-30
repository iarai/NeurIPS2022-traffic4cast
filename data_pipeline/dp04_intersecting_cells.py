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
"""Generate intersecting cells for road in the Traffic4cast Movie Grid.

This script generates lists of intersecting cells for each road segment
in the corresponding Traffic4cast Movie Grid.

Prior to starting you should have generated the road graph edge and geometry
data using the script `dp03_road_graph.py` or downloaded and extracted the
existing road graphs from the Traffic4cast competition ZIP files:
- T4C_INPUTS_2022.zip from https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip
- T4C_INPUTS_ETA_2022.zip from https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip

The script expects a subfolder `road_graph/<city>` in the passed data_folder.

The output file `cell_mapping.parquet` will be written to `road_graph/<city>`.

Usage:
  dp04_intersecting_cells.py [-h] -d DATA_FOLDER -c CITY [-f]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing a subfolder road_graph/<city> with
                        road_graph_edges.parquet and road_graph_geometries.parquet
  -c city_name, --city city_name
                        Name of the city to be processed
  -f, --force_overwrite
                        Force re-processing and overwriting existing intersecting cell files
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import geopandas
import numpy as np
import pandas
from data_helpers import get_intersecting_grid_cells
from data_helpers import get_latlon_bounds


def points_close(p1, p2):
    return np.isclose(p1[0], p2[0]) and np.isclose(p1[1], p2[1])


def is_edge_geometry_reversed(nodes_dict, u, v, line):
    line = list(line.coords)
    un = nodes_dict[u]
    un = (un["x"], un["y"])
    vn = nodes_dict[v]
    vn = (vn["x"], vn["y"])
    if points_close(line[0], un) and points_close(line[-1], vn):
        return False
    else:
        return True


def generate_intersecting_cells(city_road_graph_folder: Path, lat_min, lon_min, rotate):
    nodes_df = pandas.read_parquet(city_road_graph_folder / "road_graph_nodes.parquet")
    nodes_dict = {}
    for node_id, x, y in zip(nodes_df["node_id"], nodes_df["x"], nodes_df["y"]):
        nodes_dict[node_id] = {"x": x, "y": y}
    print(f"Loaded {len(nodes_dict)} nodes")

    edges_df = geopandas.read_parquet(city_road_graph_folder / "road_graph_edges.parquet")
    print(f"Loaded {len(edges_df)} edges")

    edges_df["geometry_reversed"] = [
        is_edge_geometry_reversed(nodes_dict, u, v, line) for u, v, line in zip(edges_df["u"], edges_df["v"], edges_df["geometry"])
    ]
    print(f'{len(edges_df[edges_df["geometry_reversed"]])} have a reversed geometry')

    edges_df["intersecting_cells"] = [
        str(get_intersecting_grid_cells(line, lon_min, lat_min, reverse=geometry_reversed, rotate=rotate))
        for line, geometry_reversed in zip(edges_df["geometry"], edges_df["geometry_reversed"])
    ]
    print(f"Done generating intersecting cells")
    return edges_df


def process_intersecting_cells(data_folder: Path, city: str, overwrite=False):
    city_road_graph_folder = data_folder / "road_graph" / city
    cells_fn = city_road_graph_folder / "road_graph_intersecting_cells.parquet"
    if not overwrite and cells_fn.exists():
        print(f"Intersecting cells file for {city} exists already. Skipping ...")
        return
    (lat_min, lat_max, lon_min, lon_max), rotate = get_latlon_bounds(city)
    cells_df = generate_intersecting_cells(city_road_graph_folder, lat_min, lon_min, rotate)
    print(f"Saving intersecting cells to {cells_fn}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
        cells_df.to_parquet(cells_fn, compression="snappy")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script creates intersecting cells for road segments.")
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
        overwrite = params["force_overwrite"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)
    process_intersecting_cells(data_folder, city=city, overwrite=overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
