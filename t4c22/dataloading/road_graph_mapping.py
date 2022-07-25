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
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from t4c22.t4c22_config import load_cc_labels
from t4c22.t4c22_config import load_inputs
from t4c22.t4c22_config import load_road_graph


class TorchRoadGraphMapping:
    def __init__(self, city: str, root: Path, df_filter, edge_attributes=None):
        self.df_filter = df_filter

        # load road graph
        df_edges, df_nodes = load_road_graph(root, city)

        # `ExternalNodeId = int64 (the osm ids)`
        # `InternalNodeId = int (0,...,num_edges-1)`

        # `edges: List[Tuple[ExternalNodeId,ExternalNodeId]]`
        self.edge_records = df_edges.to_dict("records")
        edges = [(r["u"], r["v"]) for r in self.edge_records]

        # `nodes: List[ExternalNodeId]`
        nodes = [r["node_id"] for r in df_nodes.to_dict("records")]

        # enumerate nodes and edges and create mapping
        self.node_to_int_mapping = defaultdict(lambda: -1)
        for i, k in enumerate(nodes):
            self.node_to_int_mapping[k] = i

        # edge_index: Tensor of size (2,num_edges) of InternalNodeId
        self.edge_index = torch.tensor([[self.node_to_int_mapping[n] for n, _ in edges], [self.node_to_int_mapping[n] for _, n in edges]], dtype=torch.long)

        # edge_index_d: (ExternalNodeId,ExternalNodeId) -> InternalNodeId
        self.edge_index_d = defaultdict(lambda: -1)
        self.edges = edges
        self.nodes = nodes
        for i, (u, v) in enumerate(edges):
            self.edge_index_d[(u, v)] = i

        # sanity checking edges and nodes are unique
        assert len(edges) == len(set(edges)), (len(edges), len(set(edges)))
        assert len(nodes) == len(set(nodes)), (len(nodes), len(set(nodes)))

        # sanity checking edge_index and edge_index_d size coincide with number of edges
        # beware, after accessing
        assert len(self.edge_index_d) == len(edges), (len(self.edge_index_d), len(edges))
        assert self.edge_index.size()[1] == len(edges), (self.edge_index.size()[1], len(edges))
        assert self.edge_index.size()[1] == len(self.edge_index_d), (self.edge_index.size()[1], len(self.edge_index_d))

        # sanity checking node_to_int_mapping has size number of nodes
        assert len(self.node_to_int_mapping) == len(nodes), (len(self.node_to_int_mapping), len(nodes))

        # edge_attr
        self.edge_attributes = edge_attributes
        self.edge_attr = None
        if edge_attributes is not None:
            self.edge_attr = torch.full(size=(len(self.edges), len(self.edge_attributes)), fill_value=float("nan"))
            for i, edge in enumerate(self.edge_records):

                for j, attr in enumerate(edge_attributes):
                    self.edge_attr[i, j] = edge[attr]

    def load_inputs_day_t(self, basedir: Path, city: str, split: str, day: str, t: int, idx: int) -> torch.Tensor:
        """Used by dataset getter to load input data (sparse loop counter data
        on nodes) from parquet into tensor.

        Parameters
        ----------
        basedir: data basedir see `README`
        city: "london"/"madrid"/"melbourne"
        split: "train"/"test"/...
        day: date
        t: time of day in 15-minutes in range [0,....96)
        idx: dataset index

        Returns
        -------
        Tensor of size (number-of-nodes,4).
        """
        df_x = load_inputs(basedir, city=city, split=split, day=day, df_filter=self.df_filter)

        df_x["node_id"] = df_x["node_id"].astype("int64")
        df_x = df_x.explode("volumes_1h")
        assert len(df_x) % 4 == 0
        df_x = df_x.reset_index()
        df_x["slot"] = df_x.index % 4
        df_x["volumes_1h"] = df_x["volumes_1h"].astype("float")

        x = torch.full(size=(len(self.node_to_int_mapping), 4), fill_value=float("nan"))

        # (Mis-)use (day,t) for dataloading test sets where we do not exhibit day,t
        if day == "test":
            data = df_x[(df_x["test_idx"] == idx)].copy()
        else:
            data = df_x[(df_x["day"] == day) & (df_x["t"] == t)].copy()

        data["node_index"] = [self.node_to_int_mapping[x] for x in data["node_id"]]

        # sanity check as defaultdict returns -1 for non-existing node_ids
        assert len(data[data["node_index"] < 0]) == 0
        x[data["node_index"].values, data["slot"].values] = torch.tensor(data["volumes_1h"].values).float()
        return x

    def load_cc_labels_day_t(self, basedir: Path, city: str, split: str, day: str, t: int, idx: int) -> torch.Tensor:
        """Used by dataset getter to load congestion class labels (sparse
        congestion classes) from parquet into tensor.

        Parameters
        ----------
        basedir: data basedir see `README`
        city: "london"/"madrid"/"melbourne"
        split: "train"/"test"/...
        day: date
        t: time of day in 15-minutes in range [0,....96)
        idx: dataset index


        Returns
        -------
        Float tensor of size (number-of-edges,), with edge congestion class and nan if unclassified.
        """
        df_y = load_cc_labels(basedir, city=city, split=split, day=day, with_edge_attributes=True, df_filter=self.df_filter)
        if day == "test":
            data = df_y[(df_y["test_idx"] == idx)]
        else:
            data = df_y[(df_y["day"] == day) & (df_y["t"] == t)]

        y = self._df_cc_to_torch(data)

        if len(data) == 0:
            logging.warning(f"{split} {city} {(idx, day, t)} no classified")
        return y

    def _df_cc_to_torch(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Parameters
        ----------
        data: data frame for (day,t) with columns "u", "v", "cc".

        Returns
        -------
        Float tensor of size (number-of-edges,), containing edge congestion class and nan if unclassified.
        """
        y = torch.full(size=(len(self.edges),), fill_value=float("nan"))
        if len(data[data["cc"] > 0]) > 0:
            data = data[data["cc"] > 0].copy()
            assert len(data) <= len(self.edges)
            data["edge_index"] = [self.edge_index_d[u, v] for u, v in zip(data["u"], data["v"])]

            # sanity check as defaultdict returns -1 for non-existing edges
            assert len(data[data["edge_index"] < 0]) == 0
            assert data["cc"].min() >= 1, (data["cc"].min(), data)
            assert data["cc"].max() <= 3, (data["cc"].max(), data)

            # shift left by one in tensor as model outputs only green,yellow,red but not unclassified!
            # 0 = green
            # 1 = yellow
            # 2 = red
            data["cc"] = data["cc"] - 1
            y[data["edge_index"].values] = torch.tensor(data["cc"].values).float()
        return y

    def _torch_to_df_cc(self, data: torch.Tensor, day: str, t: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        Float tensor of size (number-of-edges,3) with logits for green, yellow and red.

        Returns
        -------
        Data frame for (day,t) with columns "u", "v", "day", "t", "logit_green", "logit_yellow", "logit_red".
        """

        froms = [t[0] for t in self.edges]
        tos = [t[1] for t in self.edges]
        df = pd.concat(
            [
                pd.DataFrame(data=data[:, 0].cpu().numpy(), columns=["logit_green"]),
                pd.DataFrame(data=data[:, 1].cpu().numpy(), columns=["logit_yellow"]),
                pd.DataFrame(data=data[:, 2].cpu().numpy(), columns=["logit_red"]),
            ],
            axis=1,
        )
        df["u"] = froms
        df["v"] = tos
        df["day"] = day
        df["t"] = t
        return df
