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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing

from t4c22.dataloading.t4c22_dataset import T4c22Competitions
from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.evaluation.create_submission import create_submission_cc_plain_torch
from t4c22.evaluation.create_submission import create_submission_cc_torch_geometric
from t4c22.evaluation.create_submission import create_submission_eta_plain_torch
from t4c22.evaluation.create_submission import inference_cc_city_plain_torch_to_pandas
from t4c22.evaluation.create_submission import inference_cc_city_torch_geometric_to_pandas
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    """

    def __init__(self, in_features, out_features, hidden_features):
        super(GNN_Layer, self).__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, hidden_features), Swish(), nn.BatchNorm1d(hidden_features), nn.Linear(hidden_features, out_features), Swish()
        )
        self.update_net = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features), Swish(), nn.Linear(hidden_features, out_features), Swish())

    def forward(self, x, edge_index, batch):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i, x_j):
        """Message update."""
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message, x):  # noqa
        """Node update."""
        x += self.update_net(torch.cat((x, message), dim=-1))
        return x


class CongestioNN(torch.nn.Module):
    def __init__(self, in_features=4, out_features=32, hidden_features=32, hidden_layer=1):
        super(CongestioNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer

        # in_features have to be of the same size as out_features for the time being
        self.cgnn = torch.nn.ModuleList(modules=[GNN_Layer(self.out_features, self.out_features, self.hidden_features) for _ in range(self.hidden_layer)])

        self.head_pre_pool = nn.Sequential(  # noqa
            nn.Linear(self.out_features, self.hidden_features), Swish(), nn.Linear(self.hidden_features, self.hidden_features)
        )
        self.head_post_pool = nn.Sequential(nn.Linear(self.hidden_features, self.hidden_features), Swish(), nn.Linear(hidden_features, 1))  # noqa

        self.embedding_mlp = nn.Sequential(nn.Linear(self.in_features, self.out_features))

    def forward(self, data):
        batch = data.batch
        x = data.x
        edge_index = data.edge_index

        x = self.embedding_mlp(x)
        for i in range(self.hidden_layer):
            x = self.cgnn[i](x, edge_index, batch)

        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.swish = Swish()

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.swish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


def predict_dummy_gnn(data, device, model, predictor):
    data.x = data.x.nan_to_num(-1)
    data = data.to(device)
    h = model(data)
    x_i = torch.index_select(h, 0, data.edge_index[0])
    x_j = torch.index_select(h, 0, data.edge_index[1])
    # logits
    y_hat = predictor(x_i, x_j)
    return y_hat


class DummyUniformNN_cc(torch.nn.Module):
    def __init__(self, num_edges: int):
        super(DummyUniformNN_cc, self).__init__()
        self.num_edges = num_edges

    def forward(self, x):
        return torch.full((self.num_edges, 3), np.log(1 / 3))


class DummyNormalNN_eta(torch.nn.Module):
    def __init__(self, num_supersegments: int):
        super(DummyNormalNN_eta, self).__init__()
        self.num_edges = num_supersegments

    def forward(self, x):
        y_hat = torch.normal(torch.arange(self.num_edges).float())
        return y_hat


class DummyArangeNN_eta(torch.nn.Module):
    def __init__(self, num_supersegments: int):
        super(DummyArangeNN_eta, self).__init__()
        self.num_edges = num_supersegments

    def forward(self, x):
        y_hat = torch.arange(self.num_edges).float()
        return y_hat


class DummyInfNN_cc(torch.nn.Module):
    def __init__(self, num_edges: int):
        super(DummyInfNN_cc, self).__init__()
        self.num_edges = num_edges

    def forward(self, x):
        return torch.full((self.num_edges, 3), torch.inf)


class DummyOnesNN_cc(torch.nn.Module):
    def __init__(self, num_edges: int):
        super(DummyOnesNN_cc, self).__init__()
        self.num_edges = num_edges

    def forward(self, x):
        return torch.full((self.num_edges, 3), 1)


class DummyRandomNN_cc(torch.nn.Module):
    def __init__(self, num_edges: int):
        super(DummyRandomNN_cc, self).__init__()
        self.num_edges = num_edges

    def forward(self, x):
        return torch.log(torch.rand(self.num_edges, 3))


def apply_model_plain(data, device, model):
    x, y = data
    x = x.to(device)
    return model(x)


def apply_model_geometric(data, device, model):
    data = data.to(device)
    y_hat = model(data)
    return y_hat


def test_create_submission_cc_city_torch_geometric():
    city = "london"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=True, skip_submission=True)
        hidden_channels = 256
        num_layers = 10
        dropout = 0.0
        num_edge_classes = 3
        num_features = 4

        device = f"cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        model = CongestioNN(num_features, hidden_channels, hidden_channels, num_layers)

        # we go without checkpoint loading in unit test, add for your model:
        # model.load_state_dict(torch.load(f"../../exploration/GNN_model_{epochs:03d}.pt", map_location=device)) #noqa

        predictor = LinkPredictor(hidden_channels, hidden_channels, num_edge_classes, num_layers, dropout)

        model = model.to(device)
        predictor = predictor.to(device)

        # we go without checkpoint loading in unit test, add for your model:
        # predictor.load_state_dict(torch.load(f"../../exploration/GNN_predictor_{epochs:03d}.pt", map_location=device)) #noqa

        test_dataset = T4c22GeometricDataset(root=basedir, city=city, split="test")

        submission = inference_cc_city_torch_geometric_to_pandas(
            predict=partial(predict_dummy_gnn, device=device, model=model, predictor=predictor), test_dataset=test_dataset
        )

        print(submission)


def test_create_submission_cc_torch_geometric():
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        config = {}
        for city in cities:
            create_dummy_competition_setup(
                basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=True, skip_submission=True
            )
            hidden_channels = 256
            num_layers = 10
            dropout = 0.0
            num_edge_classes = 3
            num_features = 4

            device = f"cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)

            model = CongestioNN(num_features, hidden_channels, hidden_channels, num_layers)

            # we go without checkpoint loading in unit test, add for your model:
            # model.load_state_dict(torch.load(f"../../exploration/GNN_model_{epochs:03d}.pt", map_location=device)) #noqa

            predictor = LinkPredictor(hidden_channels, hidden_channels, num_edge_classes, num_layers, dropout)

            model = model.to(device)
            predictor = predictor.to(device)

            # we go without checkpoint loading in unit test, add for your model:
            # predictor.load_state_dict(torch.load(f"../../exploration/GNN_predictor_{epochs:03d}.pt", map_location=device)) #noqa

            test_dataset = T4c22GeometricDataset(root=basedir, city=city, split="test")

            config[city] = (test_dataset, partial(predict_dummy_gnn, device=device, model=model, predictor=predictor))

        create_submission_cc_torch_geometric(config=config, basedir=basedir, submission_name="gogogo")

        assert (basedir / "submission" / "gogogo.zip").exists()
        for city in cities:
            assert (basedir / "submission" / "gogogo" / city / "labels" / "cc_labels_test.parquet").exists()


def test_create_submission_cc_city_plain_torch():
    city = "london"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=True, skip_submission=True)

        device = f"cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        test_dataset = T4c22Dataset(root=basedir, city=city, split="test")

        model = DummyUniformNN_cc(num_edges=len(test_dataset.torch_road_graph_mapping.edges))

        submission = inference_cc_city_plain_torch_to_pandas(predict=partial(apply_model_plain, device=device, model=model), test_dataset=test_dataset)

        print(submission)


def test_create_submission_cc_plain_torch():
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        config = {}
        for city in cities:
            create_dummy_competition_setup(
                basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=True, skip_submission=True
            )

            test_dataset = T4c22Dataset(root=basedir, city=city, split="test")

            model = DummyUniformNN_cc(num_edges=len(test_dataset.torch_road_graph_mapping.edges))

            config[city] = (test_dataset, partial(apply_model_plain, device=device, model=model))

        create_submission_cc_plain_torch(config=config, basedir=basedir, submission_name="gogogo")

        assert (basedir / "submission" / "gogogo.zip").exists()
        for city in cities:
            assert (basedir / "submission" / "gogogo" / city / "labels" / "cc_labels_test.parquet").exists()


def test_create_submission_eta_plain_torch():
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        config = {}
        for city in cities:
            create_dummy_competition_setup(
                basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=True, skip_submission=True, skip_supersegments=False
            )

            test_dataset = T4c22Dataset(root=basedir, city=city, split="test", competition=T4c22Competitions.EXTENDED)

            model = DummyNormalNN_eta(num_supersegments=len(test_dataset.torch_road_graph_mapping.supersegments))

            config[city] = (test_dataset, partial(apply_model_plain, device=device, model=model))

        create_submission_eta_plain_torch(config=config, basedir=basedir, submission_name="gogogo")

        assert (basedir / "submission" / "gogogo.zip").exists()
        for city in cities:
            assert (basedir / "submission" / "gogogo" / city / "labels" / "eta_labels_test.parquet").exists()
