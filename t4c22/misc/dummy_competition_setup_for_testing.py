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
import warnings
from pathlib import Path
from typing import List
from typing import Optional

from shapely import wkb

warnings.filterwarnings("ignore", message=".*Shapely GEOS version.*")  # noqa: E402
warnings.filterwarnings("ignore", message=".*the imp module is deprecated.*")  # noqa: E402

import geopandas
import numpy as np
import pandas
from shapely.geometry import LineString

from data_pipeline.h5_helpers import write_data_to_h5
from t4c22.misc.parquet_helpers import write_df_to_parquet

NUM_SLOTS_NON_AGGREGATED = 12
NUM_SLOTS_AGGREGATED = 4
NUM_ROWS = 19
NUM_COLUMNS = 17

NUM_EDGES = 6
NUM_COUNTERS = 3
NUM_NODES = 4
NUM_SUPERSEGMENTS = 2

EDGES = [
    {
        "u": 10495890,
        "v": 457298457598,
        "osmid": "1",
        "maxspeed": "50",
        "parsed_maxspeed": 50.0,
        "speed_kph": 50.0,
        "importance": 1,
        "highway": "tertiary",
        "oneway": False,
        "lanes": 2,
        "tunnel": "",
        "length_meters": 47.84958973,
        "counter_distance": 2,
    },
    {
        "u": 10495890,
        "v": 9824598274857,
        "osmid": "2",
        "maxspeed": "60",
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
    {
        "u": 457298457598,
        "v": 10495890,
        "osmid": "3",
        "maxspeed": "50",
        "parsed_maxspeed": 50.0,
        "speed_kph": 50.0,
        "importance": 1,
        "highway": "tertiary",
        "oneway": False,
        "lanes": 2,
        "tunnel": "",
        "length_meters": 47.84958973,
        "counter_distance": 2,
    },
    {
        "u": 457298457598,
        "v": 9824598274857,
        "osmid": "4",
        "maxspeed": "30",
        "parsed_maxspeed": 30.0,
        "speed_kph": 30.0,
        "importance": 1,
        "highway": "residential",
        "oneway": False,
        "lanes": 1,
        "tunnel": "",
        "length_meters": 831.456,
        "counter_distance": 4,
    },
    {
        "u": 9824598274857,
        "v": 457298457598,
        "osmid": "5",
        "maxspeed": "30",
        "parsed_maxspeed": 30.0,
        "speed_kph": 30.0,
        "importance": 1,
        "highway": "residential",
        "oneway": False,
        "lanes": 1,
        "tunnel": "",
        "length_meters": 831.456,
        "counter_distance": 4,
    },
    {
        "u": 9824598274857,
        "v": 89234759823745,
        "osmid": "6",
        "maxspeed": "30",
        "parsed_maxspeed": 30.0,
        "speed_kph": 30.0,
        "importance": 1,
        "highway": "residential",
        "oneway": False,
        "lanes": 1,
        "tunnel": "",
        "length_meters": 831.456,
        "counter_distance": 4,
    },
]

NODES = [
    {
        "node_id": 10495890,
        "counter_info": "ABCD",
        "num_assigned": "3",
        "x": -3.684,
        "y": 51.698,
    },
    {
        "node_id": 457298457598,
        "counter_info": "XYZ",
        "num_assigned": "3",
        "x": -3.688,
        "y": 51.697,
    },
    {
        "node_id": 9824598274857,
        "counter_info": "FFFFF",
        "num_assigned": "1",
        "x": -3.679,
        "y": 51.688,
    },
    {
        "node_id": 89234759823745,
        "counter_info": "",
        "num_assigned": "",
        "x": -3.680,
        "y": 51.690,
    },
]

SUPERSEGMENTS = [
    {
        "identifier": "10495890,9824598274857",
        "nodes": [
            10495890,
            457298457598,
            9824598274857,
        ],
    },
    {
        "identifier": "9824598274857,10495890",
        "nodes": [
            9824598274857,
            457298457598,
            10495890,
        ],
    },
]

INTERSECTING_CELLS = {
    (10495890, 457298457598): [(2, 2, 3, 0.25), (2, 3, 3, 0.25), (1, 6, 3, 0.0), (3, 1, 3, 0.0), (2, 4, 3, 0.25), (2, 5, 3, 0.25)],
    (10495890, 9824598274857): [
        (8, 9, 2, 0.1),
        (5, 8, 2, 0.0),
        (9, 9, 2, 0.1),
        (2, 6, 2, 0.1),
        (11, 11, 2, 0.0),
        (10, 9, 2, 0.0),
        (6, 7, 2, 0.0),
        (12, 10, 2, 0.0),
        (1, 6, 2, 0.0),
        (7, 9, 2, 0.0),
        (4, 7, 2, 0.1),
        (5, 7, 2, 0.1),
        (9, 10, 2, 0.0),
        (6, 8, 2, 0.1),
        (10, 10, 2, 0.1),
        (7, 8, 2, 0.1),
        (2, 5, 2, 0.0),
        (11, 10, 2, 0.1),
        (3, 6, 2, 0.1),
    ],
    (457298457598, 10495890): [(1, 6, 0, 0.0), (3, 1, 0, 0.0), (2, 4, 0, 0.25), (2, 5, 0, 0.25), (2, 2, 0, 0.25), (2, 3, 0, 0.25)],
    (457298457598, 9824598274857): [
        (5, 4, 2, 0.1),
        (2, 2, 2, 0.0),
        (9, 9, 2, 0.0),
        (6, 5, 2, 0.1),
        (11, 11, 2, 0.0),
        (3, 1, 2, 0.0),
        (7, 5, 2, 0.0),
        (10, 9, 2, 0.1),
        (4, 3, 2, 0.1),
        (12, 10, 2, 0.0),
        (9, 8, 2, 0.1),
        (10, 8, 2, 0.0),
        (5, 5, 2, 0.0),
        (6, 4, 2, 0.0),
        (7, 6, 2, 0.1),
        (11, 10, 2, 0.1),
        (6, 6, 2, 0.0),
        (3, 2, 2, 0.1),
        (8, 7, 2, 0.1),
    ],
    (9824598274857, 457298457598): [
        (7, 6, 1, 0.1),
        (11, 10, 1, 0.1),
        (4, 4, 1, 0.0),
        (3, 2, 1, 0.1),
        (9, 7, 1, 0.0),
        (8, 7, 1, 0.1),
        (5, 4, 1, 0.1),
        (2, 2, 1, 0.0),
        (6, 5, 1, 0.1),
        (11, 11, 1, 0.0),
        (3, 1, 1, 0.0),
        (10, 9, 1, 0.1),
        (4, 3, 1, 0.1),
        (7, 7, 1, 0.0),
        (8, 6, 1, 0.0),
        (5, 3, 1, 0.0),
        (8, 8, 1, 0.0),
        (12, 10, 1, 0.0),
        (9, 8, 1, 0.1),
    ],
    (9824598274857, 89234759823745): [(7, 6, 1, 0.1)],
}


def create_dummy_competition_setup(  # noqa:C901
    basedir: Path,
    city: str,
    train_dates: List[str],
    num_test_slots=20,
    skip_road_graph: bool = False,
    skip_speed_classes: bool = False,
    skip_train_labels: bool = False,
    skip_golden: bool = False,
    test_dates: Optional[List[str]] = None,
    skip_movie: bool = True,
    skip_movie_15min: bool = True,
    skip_submission: bool = True,
    skip_tests: bool = False,
    skip_supersegments: bool = False,
    skip_intersecting_cells: bool = True,
    skip_speed_clusters: bool = True,
    skip_free_flow: bool = True,
    seed: int = 666,
    num_movie_channels=8,
):
    np.random.seed(seed=seed)

    # `road_graph`
    assert len(EDGES) == NUM_EDGES, (len(EDGES), NUM_EDGES)
    assert len(NODES) == NUM_NODES, (len(NODES), NUM_NODES)

    if not skip_road_graph:
        node_points = {}
        for n in NODES:
            node_points[n["node_id"]] = (n["x"], n["y"])
        edges = []
        for ed in EDGES:
            ed = ed.copy()
            ed["geometry"] = LineString([node_points[ed["u"]], node_points[ed["v"]]])
            edges.append(ed)
        edges_parquet = basedir / "road_graph" / city / "road_graph_edges.parquet"
        edges_parquet.parent.mkdir(parents=True, exist_ok=True)
        edges_df = geopandas.GeoDataFrame.from_records(edges)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
            edges_df.to_parquet(edges_parquet, compression="snappy")

        ic_df = edges_df.copy()
        ic_df["intersecting_cells"] = [str(INTERSECTING_CELLS[(u, v)]) for u, v in zip(ic_df["u"], ic_df["v"])]
        if not skip_intersecting_cells:
            ic_parquet = basedir / "road_graph" / city / f"road_graph_intersecting_cells.parquet"
            ic_parquet.parent.mkdir(parents=True, exist_ok=True)
            ic_df.to_parquet(ic_parquet, compression="snappy")

        ff_df = ic_df.copy()
        # Create test free flow between 20 and 70
        ff_df["free_flow_kph"] = [20 + np.random.rand() * 50 for _ in range(len(ff_df))]
        ff_df["gkey"] = [hash(wkb.dumps(g)) for g in ff_df["geometry"]]
        if not skip_free_flow:
            ff_parquet = basedir / "road_graph" / city / "road_graph_freeflow.parquet"
            ff_parquet.parent.mkdir(parents=True, exist_ok=True)
            ff_df.to_parquet(ff_parquet, compression="snappy")

        nodes_parquet = basedir / "road_graph" / city / "road_graph_nodes.parquet"
        write_df_to_parquet(
            pandas.DataFrame.from_records(NODES),
            fn=nodes_parquet,
        )

    if not skip_supersegments:
        supersegments_parquet = basedir / "road_graph" / city / "road_graph_supersegments.parquet"
        assert len(SUPERSEGMENTS) == NUM_SUPERSEGMENTS, (len(SUPERSEGMENTS), NUM_SUPERSEGMENTS)
        write_df_to_parquet(
            pandas.DataFrame.from_records(SUPERSEGMENTS),
            fn=supersegments_parquet,
        )

    if not skip_speed_clusters:
        speed_clusters_h5 = basedir / "movie_speed_clusters" / city / f"speed_clusters.h5"
        speed_clusters_h5.parent.mkdir(exist_ok=True, parents=True)
        row_data = []
        for _ in range(NUM_ROWS):
            col_data = []
            for _ in range(NUM_COLUMNS):
                heading_data = []
                for _ in range(4):
                    cluster_data = []
                    for _ in range(5):
                        speed = 10 + np.random.rand() * 14  # Test clusters are between 10 and 24
                        vol = np.random.randint(1, 25)
                        cluster_data.append([speed, vol])
                    heading_data.append(cluster_data)
                col_data.append(heading_data)
            row_data.append(col_data)
        data = np.array(row_data)
        write_data_to_h5(filename=speed_clusters_h5, data=data, dtype=np.uint8)

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
                    for node in NODES
                ]
            ),
            fn=counters_parquet,
        )

    if not skip_speed_classes:
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
                        for edge in EDGES
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
                        for edge in EDGES
                    ]
                ),
                fn=cc_labels_parquet,
            )
            if not skip_supersegments:
                eta_labels_parquet = basedir / "train" / city / "labels" / f"eta_labels_{train_date}.parquet"
                eta_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
                write_df_to_parquet(
                    pandas.DataFrame.from_records(
                        [
                            {
                                "identifier": supersegment["identifier"],
                                "day": train_date,
                                "t": t,
                                "eta": np.clip(np.random.normal(loc=30, scale=10), a_min=2, a_max=None),
                            }
                            for t in range(96)
                            for supersegment in SUPERSEGMENTS
                        ]
                    ),
                    fn=eta_labels_parquet,
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
                    for node in NODES
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
                    for edge in EDGES
                ]
            ),
            fn=cc_labels_parquet,
        )
        if not skip_supersegments:
            eta_labels_parquet = basedir / "withheld" / "golden" / city / "labels" / f"eta_labels_test.parquet"
            eta_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
            write_df_to_parquet(
                pandas.DataFrame.from_records(
                    [
                        {
                            "identifier": supersegment["identifier"],
                            "test_idx": test_idx,
                            "eta": np.clip(np.random.normal(loc=30, scale=10), a_min=2, a_max=None),
                        }
                        for test_idx in range(num_test_slots)
                        for supersegment in SUPERSEGMENTS
                    ]
                ),
                fn=eta_labels_parquet,
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
                        for edge in EDGES
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
                        for node in NODES
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
                    for edge in EDGES
                ]
            ),
            fn=cc_labels_parquet,
        )

    # movie
    if not skip_movie:
        for train_date in train_dates:

            movie_h5 = basedir / "movie" / city / f"{train_date}_{city}_{num_movie_channels}ch.h5"
            movie_h5.parent.mkdir(exist_ok=True, parents=True)

            write_data_to_h5(
                filename=movie_h5, data=np.random.randint(256, size=(NUM_SLOTS_NON_AGGREGATED, NUM_ROWS, NUM_COLUMNS, num_movie_channels), dtype=np.uint8)
            )

    # movie
    if not skip_movie_15min:
        for train_date in train_dates:
            movie_h5 = basedir / "movie_15min" / city / f"{train_date}_{city}_8ch_15min.h5"
            movie_h5.parent.mkdir(exist_ok=True, parents=True)
            data = []
            for _ in range(NUM_SLOTS_AGGREGATED):
                row_data = []
                for _ in range(NUM_ROWS):
                    col_data = []
                    for _ in range(NUM_COLUMNS):
                        cell_data = []
                        for _ in range(0, 8, 2):
                            speed = np.random.randint(0, 256)
                            if speed > 0:
                                cell_data.append(np.random.randint(1, 256))
                                cell_data.append(speed)
                            else:
                                cell_data.append(0)
                                cell_data.append(0)
                        col_data.append(cell_data)
                    row_data.append(col_data)
                data.append(row_data)
            data = np.array(data)
            write_data_to_h5(filename=movie_h5, data=data, dtype=np.uint8)
