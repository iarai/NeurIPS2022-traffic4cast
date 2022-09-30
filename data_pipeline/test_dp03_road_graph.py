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

import data_helpers as dp_data_helpers
import pandas
from dp03_road_graph import main

import t4c22.misc.dummy_competition_setup_for_testing as dcsft
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


GRAPHML_HEADER = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d25" for="edge" attr.name="width" attr.type="string"/>
  <key id="d24" for="edge" attr.name="est_width" attr.type="string"/>
  <key id="d23" for="edge" attr.name="tunnel" attr.type="string"/>
  <key id="d22" for="edge" attr.name="access" attr.type="string"/>
  <key id="d21" for="edge" attr.name="junction" attr.type="string"/>
  <key id="d20" for="edge" attr.name="bridge" attr.type="string"/>
  <key id="d19" for="edge" attr.name="lanes" attr.type="string"/>
  <key id="d18" for="edge" attr.name="travel_time" attr.type="string"/>
  <key id="d17" for="edge" attr.name="speed_kph" attr.type="string"/>
  <key id="d16" for="edge" attr.name="geometry" attr.type="string"/>
  <key id="d15" for="edge" attr.name="length" attr.type="string"/>
  <key id="d14" for="edge" attr.name="oneway" attr.type="string"/>
  <key id="d13" for="edge" attr.name="maxspeed" attr.type="string"/>
  <key id="d12" for="edge" attr.name="highway" attr.type="string"/>
  <key id="d11" for="edge" attr.name="name" attr.type="string"/>
  <key id="d10" for="edge" attr.name="ref" attr.type="string"/>
  <key id="d9" for="edge" attr.name="osmid" attr.type="string"/>
  <key id="d8" for="node" attr.name="ref" attr.type="string"/>
  <key id="d7" for="node" attr.name="street_count" attr.type="string"/>
  <key id="d6" for="node" attr.name="highway" attr.type="string"/>
  <key id="d5" for="node" attr.name="x" attr.type="string"/>
  <key id="d4" for="node" attr.name="y" attr.type="string"/>
  <key id="d3" for="graph" attr.name="simplified" attr.type="string" />
  <key id="d2" for="graph" attr.name="crs" attr.type="string" />
  <key id="d1" for="graph" attr.name="created_with" attr.type="string" />
  <key id="d0" for="graph" attr.name="created_date" attr.type="string" />
  <graph edgedefault="directed">
  <data key="d1">OSMnx mock for testing</data>
  <data key="d2">epsg:4326</data>
  <data key="d3">True</data>

"""
GRAPHML_FOOTER = """
  </graph>
</graphml>
"""


def test_road_graph():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        dp_data_helpers.T4C_BBOXES = {
            "london": {"bounds": [5167600, 5170000, -369000, -367000]},
        }
        dp_data_helpers.NUM_ROWS = 24
        dp_data_helpers.NUM_COLUMNS = 20

        create_dummy_competition_setup(basedir=tmp_dir, city="london", train_dates=[], skip_train_labels=True, skip_road_graph=True, skip_supersegments=True)

        nodes_pq = tmp_dir / "road_graph" / "london" / "road_graph_nodes.parquet"
        edges_pq = tmp_dir / "road_graph" / "london" / "road_graph_edges.parquet"
        assert not nodes_pq.exists()
        assert not edges_pq.exists()

        graphml_fn = tmp_dir / "road_graph" / "london" / "road_graph.graphml"
        assert not graphml_fn.exists()
        node_points = {}
        graph_elements = []
        for n in dcsft.NODES:
            nid = n["node_id"]
            x = n["x"]
            y = n["y"]
            graph_elements.append(f'  <node id="{nid}"><data key="d4">{y}</data><data key="d5">{x}</data></node>')
            node_points[n["node_id"]] = f'{n["x"]} {n["y"]}'
        for i, e in enumerate(dcsft.EDGES):
            u = e["u"]
            v = e["v"]
            highway = e["highway"]  # d12
            maxspeed = str(e["parsed_maxspeed"])  # d13
            oneway = e["oneway"]  # d14
            length = e["length_meters"]  # d15
            geometry = f"LINESTRING ({node_points[u]}, {node_points[v]})"  # d16
            speed_kph = e["speed_kph"]  # d17
            lanes = e["lanes"]  # d19
            graph_elements.append(
                f'  <edge source="{u}" target="{v}" id="0"><data key="d9">{i}</data><data key="d12">{highway}</data>'
                f'<data key="d13">{maxspeed}</data><data key="d14">{oneway}</data><data key="d15">{length}</data>'
                f'<data key="d16">{geometry}</data><data key="d17">{speed_kph}</data><data key="d19">{lanes}</data>'
                f'<data key="d23">False</data></edge>'
            )
        graphml_str = GRAPHML_HEADER + "\n".join(graph_elements) + GRAPHML_FOOTER
        graphml_fn.parent.mkdir(parents=True, exist_ok=True)
        with open(graphml_fn, "w") as f:
            f.write(graphml_str)

        main(["-d", str(tmp_dir), "-c", "london"])

        assert nodes_pq.exists()
        assert edges_pq.exists()

        nodes_df = pandas.read_parquet(nodes_pq)
        assert len(nodes_df) == 4
        assert 10495890 in list(nodes_df["node_id"])

        edges_df = pandas.read_parquet(edges_pq)
        assert len(edges_df) == 6
        assert 10495890 in list(edges_df["u"])
        assert 10495890 in list(edges_df["v"])
