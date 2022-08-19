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
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pandas

from t4c22.misc.movie_aggregation import write_data_to_h5
from t4c22.misc.parquet_helpers import write_df_to_parquet

NUM_SLOTS_NON_AGGREGATED = 12
NUM_SLOTS_AGGREGATED = 4
NUM_ROWS = 19
NUM_COLUMNS = 17


def create_dummy_competition_setup(  # noqa:C901
    basedir: Path,
    city: str,
    train_dates: List[str],
    num_test_slots=20,
    skip_train_labels: bool = False,
    skip_golden: bool = False,
    test_dates: Optional[List[str]] = None,
    skip_movie: bool = True,
    skip_submission: bool = True,
    skip_tests: bool = False,
):

    # `road_graph`
    edges = [
        {
            "u": 10495890,
            "v": 457298457598,
            "parsed_maxspeed": 50.0,
            "speed_kph": 50.0,
            "importance": 1,
            "highway": "tertiary",
            "oneway": True,
            "lanes": 4,
            "tunnel": "",
            "length_meters": 47.84958973,
            "counter_distance": 2,
        },
        {
            "u": 10495890,
            "v": 9824598274857,
            "parsed_maxspeed": 60.0,
            "speed_kph": 55.0,
            "importance": 1,
            "highway": "tertiary",
            "oneway": True,
            "lanes": 3,
            "tunnel": "",
            "length_meters": 47.234523,
            "counter_distance": 3,
        },
    ]
    edges_parquet = basedir / "road_graph" / city / "road_graph_edges.parquet"
    edges_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_df_to_parquet(
        pandas.DataFrame.from_records(edges),
        fn=edges_parquet,
    )
    nodes_parquet = basedir / "road_graph" / city / "road_graph_nodes.parquet"
    nodes = [
        {
            "node_id": 10495890,
            "counter_info": "ABCD",
            "num_assigned": "3",
            "x": -3.9847589475,
            "y": 40.2987459827,
        },
        {
            "node_id": 457298457598,
            "counter_info": "XYZ",
            "num_assigned": "3",
            "x": -3.9847589475,
            "y": 40.2987459827,
        },
        {
            "node_id": 9824598274857,
            "counter_info": "FFFFF",
            "num_assigned": "1",
            "x": -3.48574895,
            "y": 40.9483728957,
        },
    ]
    write_df_to_parquet(
        pandas.DataFrame.from_records(nodes),
        fn=nodes_parquet,
    )

    # `train/<city>/input/counters_<date>.parquet`
    for train_date in train_dates:
        counters_parquet = basedir / "train" / city / "input" / f"counters_{train_date}.parquet"
        counters_parquet.parent.mkdir(parents=True, exist_ok=True)
        write_df_to_parquet(
            pandas.DataFrame.from_records(
                [
                    {
                        "node_id": node["node_id"],
                        "day": train_date,
                        "t": t,
                        "volumes_1h": list(np.random.random(4) * 500),
                    }
                    for t in range(96)
                    for node in nodes
                ]
            ),
            fn=counters_parquet,
        )

    # `speed_classes/`
    for train_date in train_dates:
        speed_classes_parquet = basedir / "speed_classes" / city / f"speed_classes_{train_date}.parquet"
        speed_classes_parquet.parent.mkdir(parents=True, exist_ok=True)
        write_df_to_parquet(
            pandas.DataFrame.from_records(
                [
                    {
                        "u": edge["u"],
                        "v": edge["v"],
                        "day": train_date,
                        "t": t,
                        # 1,...,5 (high is exclusive!)
                        "volume_class": np.random.randint(1, 6),
                        "median_speed_kph": np.random.random() * 120,
                        "free_flow_kph": np.random.random() * 120,
                    }
                    for t in range(96)
                    for edge in edges
                ]
            ),
            fn=speed_classes_parquet,
        )

    # `train/<city>/labels/cc_labels_<date>.parquet`
    if not skip_train_labels:
        for train_date in train_dates:
            cc_labels_parquet = basedir / "train" / city / "labels" / f"cc_labels_{train_date}.parquet"
            cc_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
            write_df_to_parquet(
                pandas.DataFrame.from_records(
                    [
                        {
                            "u": edge["u"],
                            "v": edge["v"],
                            "day": train_date,
                            "t": t,
                            # 0,1,2,3 (high is exclusive!)
                            "cc": np.random.randint(0, 4),
                        }
                        for t in range(96)
                        for edge in edges
                    ]
                ),
                fn=cc_labels_parquet,
            )

    if not skip_tests:
        counters_parquet = basedir / "test" / city / "input" / f"counters_test.parquet"
        counters_parquet.parent.mkdir(parents=True, exist_ok=True)

        write_df_to_parquet(
            pandas.DataFrame.from_records(
                [
                    {
                        "node_id": node["node_id"],
                        "test_idx": test_idx,
                        "volumes_1h": list(np.random.random(4) * 500),
                    }
                    for test_idx in range(num_test_slots)
                    for node in nodes
                ]
            ),
            fn=counters_parquet,
        )

    if not skip_golden:
        cc_labels_parquet = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
        cc_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
        write_df_to_parquet(
            pandas.DataFrame.from_records(
                [
                    {
                        "u": edge["u"],
                        "v": edge["v"],
                        "test_idx": test_idx,
                        # 0,1,2,3 (high is exclusive!)
                        "cc": np.random.randint(0, 4),
                    }
                    for test_idx in range(num_test_slots)
                    for edge in edges
                ]
            ),
            fn=cc_labels_parquet,
        )
    if test_dates is not None:
        for test_date in test_dates:
            cc_labels_parquet = basedir / "withheld" / "test" / city / "labels" / f"cc_labels_{test_date}.parquet"
            cc_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
            write_df_to_parquet(
                pandas.DataFrame.from_records(
                    [
                        {
                            "u": edge["u"],
                            "v": edge["v"],
                            "day": train_dates,
                            "t": t,
                            # 0,1,2,3 (high is exclusive!)
                            "cc": np.random.randint(0, 4),
                        }
                        for t in range(96)
                        for edge in edges
                    ]
                ),
                fn=cc_labels_parquet,
            )
            counters_parquet = basedir / "withheld" / "test" / city / "input" / f"counters_{test_date}.parquet"
            counters_parquet.parent.mkdir(parents=True, exist_ok=True)
            write_df_to_parquet(
                pandas.DataFrame.from_records(
                    [
                        {
                            "node_id": node["node_id"],
                            "day": test_date,
                            "t": t,
                            "volumes_1h": list(np.random.random(4) * 500),
                        }
                        for t in range(96)
                        for node in nodes
                    ]
                ),
                fn=counters_parquet,
            )

    if not skip_submission:
        cc_labels_parquet = basedir / "submission" / city / "labels" / f"cc_labels_test.parquet"
        cc_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
        write_df_to_parquet(
            pandas.DataFrame.from_records(
                [
                    {
                        "u": edge["u"],
                        "v": edge["v"],
                        "test_idx": test_idx,
                        "logit_green": np.random.random() * (-5),
                        "logit_yellow": np.random.random() * (-5),
                        "logit_red": np.random.random() * (-5),
                    }
                    for test_idx in range(num_test_slots)
                    for edge in edges
                ]
            ),
            fn=cc_labels_parquet,
        )

    # movie
    if not skip_movie:
        for train_date in train_dates:
            movie_h5 = basedir / "movie" / city / f"{train_date}_{city}_8ch.h5"
            movie_h5.parent.mkdir(exist_ok=True, parents=True)
            write_data_to_h5(filename=movie_h5, data=np.random.randint(256, size=(NUM_SLOTS_NON_AGGREGATED, NUM_ROWS, NUM_COLUMNS, 8), dtype=np.uint8))
