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
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from t4c22.misc.parquet_helpers import load_df_from_parquet
from t4c22.misc.parquet_helpers import write_df_to_parquet


def add_node_attributes_to_edges_parquet(basedir: Path, city: str):
    df_nodes: pd.DataFrame = load_df_from_parquet(basedir / "road_graph" / city / "road_graph_nodes.parquet")
    edges_parquet = basedir / "road_graph" / city / "road_graph_edges.parquet"
    df_edges: pd.DataFrame = load_df_from_parquet(edges_parquet)
    expected_len = len(df_edges)
    logging.info(f"columns after %s: %s", city, df_edges.columns)
    assert len(set(df_edges.columns).intersection(set(df_nodes.columns))) == 0, (df_edges.columns, df_nodes.columns)

    df_edges = df_edges.merge(df_nodes, left_on=["u"], right_on=["node_id"], how="left", suffixes=("", "_u"))
    # suffixes are applied only on conflict...
    df_edges.rename(columns={col: f"{col}_u" for col in df_nodes.columns}, inplace=True)
    # suffixes are applied only on conflict...
    df_edges = df_edges.merge(df_nodes, left_on=["v"], right_on=["node_id"], how="left", suffixes=("", "_v"))
    df_edges.rename(columns={col: f"{col}_v" for col in df_nodes.columns}, inplace=True)
    logging.info(f"columns after %s: %s", city, df_edges.columns)

    assert len(df_edges) == expected_len, (len(df_edges), expected_len)

    write_df_to_parquet(df=df_edges, fn=edges_parquet)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("This script takes the data folder as input and adds to the node attributes to the edges file."))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=Path,
        help="Folder containing T4c22 data",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--city",
        type=Path,
        help="City. If not passed, all three cities will be processed",
        required=True,
    )
    return parser


def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(argv)
        params = vars(params)

        cities = ["london", "madrid", "melbourne"]
        if "city" in params:
            cities = [params["city"]]
        for city in cities:
            add_node_attributes_to_edges_parquet(basedir=params["data_folder"], city=city)

    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
