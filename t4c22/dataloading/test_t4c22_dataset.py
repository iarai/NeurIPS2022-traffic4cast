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
from functools import partial
from pathlib import Path

import pytest
import torch
import tqdm

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.evaluation.create_submission import inference_cc_city_plain_torch_to_pandas
from t4c22.evaluation.test_create_submission import DummyRandomNN
from t4c22.misc.add_position_to_edges import add_node_attributes_to_edges_parquet
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.misc.parquet_helpers import load_df_from_parquet


@pytest.mark.parametrize(
    "dataset_class,extractor,edge_attributes,use_cachedir",
    [
        (T4c22GeometricDataset, lambda data: (data.x, data.y), None, True),
        (T4c22GeometricDataset, lambda data: (data.x, data.y), None, False),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance"], True),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance"], False),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance", "x_u", "x_v", "y_u", "y_v"], True),
        (T4c22GeometricDataset, lambda data: (data.x, data.y, data.edge_attr), ["importance", "x_u", "x_v", "y_u", "y_v"], False),
        (T4c22Dataset, lambda data: data, None, True),
        (T4c22Dataset, lambda data: data, None, False),
        (T4c22Dataset, lambda data: data, ["importance"], True),
        (T4c22Dataset, lambda data: data, ["importance"], False),
        (T4c22Dataset, lambda data: data, ["importance", "x_u", "x_v", "y_u", "y_v"], True),
        (T4c22Dataset, lambda data: data, ["importance", "x_u", "x_v", "y_u", "y_v"], False),
    ],
)
def test_T4c22Dataset(dataset_class, extractor, edge_attributes, use_cachedir):  # noqa:C901
    city = "gotham"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        with tempfile.TemporaryDirectory() as cachedir:
            basedir = Path(basedir)
            cachedir = Path(cachedir)
            if not use_cachedir:
                cachedir = None

            create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots)
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


def test_create_submission_cc_city_plain_torch():
    """Test torch -> pandas -> torch works correctly, i.e. going through pandas
    does not change."""
    # TODO same with geometric?
    city = "gotham"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)

        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[], test_dates=[date], num_test_slots=num_test_slots)

        ds = T4c22Dataset(root=basedir, city=city, split="test")

        seed = 666

        model = DummyRandomNN(num_edges=len(ds.torch_road_graph_mapping.edges))

        predict, y_hats_torch = _inference_torch(ds, model, seed)

        torch.manual_seed(seed)
        df = inference_cc_city_plain_torch_to_pandas(test_dataset=ds, predict=predict)
        y_hats_torch_pandas_torch = torch.from_numpy(df[["logit_green", "logit_yellow", "logit_red"]].to_numpy())

        assert torch.allclose(y_hats_torch_pandas_torch, y_hats_torch)


def _inference_torch(ds, model, seed=None):
    def predict(data, device, model):
        x, y = data
        x = x.to(device)
        return model(x)

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    predict = partial(predict, device=device, model=model)
    if seed is not None:
        torch.manual_seed(seed)
    y_hats_torch = []
    for _, data in tqdm.tqdm(enumerate(ds), total=len(ds)):
        y_hat = predict(data)
        y_hats_torch.append(y_hat)
        x, y = data
    y_hats_torch = torch.cat(y_hats_torch)
    return predict, y_hats_torch
