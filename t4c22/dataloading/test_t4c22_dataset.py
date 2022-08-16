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
import tempfile
from pathlib import Path

import pytest
import torch
import tqdm

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.misc.add_position_to_edges import add_node_attributes_to_edges_parquet
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.misc.parquet_helpers import load_df_from_parquet


@pytest.mark.parametrize(
    "dataset_class,extractor,edge_attributes",
    [
        (T4c22GeometricDataset, lambda data: (data.x, data.y), None),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance"]),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance", "x_u", "x_v", "y_u", "y_v"]),
        (T4c22Dataset, lambda data: data, None),
        (T4c22Dataset, lambda data: data, ["importance"]),
        (T4c22Dataset, lambda data: data, ["importance", "x_u", "x_v", "y_u", "y_v"]),
    ],
)
def test_T4c22Dataset(dataset_class, extractor, edge_attributes):  # noqa:C901
    city = "gotham"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        with tempfile.TemporaryDirectory() as cachedir:
            basedir = Path(basedir)
            cachedir = Path(cachedir)

            create_dummy_competition_setup(basedir=basedir, city=city, date=date, num_test_slots=num_test_slots)
            if edge_attributes is not None and "x_u" in edge_attributes:
                add_node_attributes_to_edges_parquet(basedir=basedir, city=city)

            ds = dataset_class(root=basedir, city=city, split="train", cachedir=cachedir, edge_attributes=edge_attributes)
            for idx, data in tqdm.tqdm(enumerate(ds), total=len(ds)):
                if edge_attributes is None:
                    x, y = extractor(data)
                else:
                    x, y, edge_attr = extractor(data)
                day, t = ds.day_t[idx]

                # --------------------------------------------
                #  edges
                # --------------------------------------------
                # reload congestion classes from parquet
                fn = basedir / "train" / city / "labels" / f"cc_labels_{day}.parquet"
                df = load_df_from_parquet(fn)

                df = df[(df["day"] == day) & (df["t"] == t)]
                assert df.groupby(["u", "v"]).count()["cc"].min() == 1
                assert df.groupby(["u", "v"]).count()["cc"].max() == 1

                for (u, v), y_actual in zip(ds.torch_road_graph_mapping.edges, y):
                    df_ = df[(df["u"] == u) & (df["v"] == v)]
                    assert len(df_) == 1, df_
                    y_expected = df_.iloc[0]["cc"]

                    if torch.isnan(y_actual):
                        assert y_expected == 0
                    else:
                        assert y_expected == y_actual + 1, df_

                # --------------------------------------------
                #  nodes
                # --------------------------------------------
                # reload congestion classes from parquet
                fn = basedir / "train" / city / "input" / f"counters_{day}.parquet"
                df = load_df_from_parquet(fn)

                df = df[(df["day"] == day) & (df["t"] == t)]
                for node_id, x_actual in tqdm.tqdm(zip(ds.torch_road_graph_mapping.nodes, x)):
                    df_ = df[(df["node_id"] == node_id)]
                    assert len(df_) == 1, df_
                    df_ = df_.explode("volumes_1h")

                    for t in range(4):
                        x_expected = df_.iloc[t]["volumes_1h"]
                        if torch.isnan(x_actual[t]):
                            assert x_expected == 0
                        else:
                            assert x_expected == x_actual[t]
                # --------------------------------------------
                #  edge_attr
                # --------------------------------------------
                if edge_attributes is not None:
                    expected_size = (len(ds.torch_road_graph_mapping.edges), len(edge_attributes))
                    assert edge_attr.size() == expected_size, (expected_size, edge_attr.size())

            ds = dataset_class(root=basedir, city=city, split="test", cachedir=cachedir, day_t_filter=None)
            assert len(ds) == num_test_slots
            for _, data in tqdm.tqdm(enumerate(ds), total=len(ds)):
                _ = extractor(data)


def test_torch_to_df_to_torch():
    # TODO we want to be sure evaluation is the same if evaluation directly on the catted tensors or after writing to parquet and evaluation the submission!
    # TODO test torch -> pandas -> torch gives the same
    # TODO test torch -> crossentropy is the same as torch -> pandas -> toch crossentropy
    pass
