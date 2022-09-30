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

import numpy as np
import pandas
from dp05_free_flow import get_free_flow
from dp05_free_flow import main
from dp05_free_flow import merge_speed_clusters_from_intersecting_cells

from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


def test_free_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        create_dummy_competition_setup(
            basedir=tmp_dir, city="london", train_dates=[], skip_train_labels=True, skip_speed_clusters=False, skip_intersecting_cells=False
        )
        assert (tmp_dir / "road_graph" / "london" / "road_graph_edges.parquet").exists()
        assert (tmp_dir / "road_graph" / "london" / "road_graph_intersecting_cells.parquet").exists()

        free_flow_pq = tmp_dir / "road_graph" / "london" / "road_graph_freeflow.parquet"
        assert not free_flow_pq.exists()

        main(["-d", str(tmp_dir), "-c", "london"])

        assert free_flow_pq.exists()
        free_flow_df = pandas.read_parquet(free_flow_pq)
        assert len(free_flow_df) == 6
        assert free_flow_df["free_flow_kph"].min() > 0
        assert free_flow_df["free_flow_kph"].max() < 120


def test_get_free_flow():
    merged_speed_clusters = np.array(
        sorted([[5.0, 10], [15.0, 20], [25.0, 30], [35.0, 40], [45.0, 5], [10.0, 15], [20.0, 25], [30.0, 35], [40.0, 30], [50.0, 3]])
    )
    ff = get_free_flow(merged_speed_clusters, speed_limit=-1, volume_threshold=0.2)
    assert ff == 35.0
    ff = get_free_flow(merged_speed_clusters, speed_limit=-1, volume_threshold=0.1)
    assert ff == 40.0
    ff = get_free_flow(merged_speed_clusters, speed_limit=-1, volume_threshold=0.5)
    assert ff == 30.0
    ff = get_free_flow(merged_speed_clusters, speed_limit=30, volume_threshold=0.2)
    assert ff == 35.0
    ff = get_free_flow(merged_speed_clusters, speed_limit=50, volume_threshold=0.2)
    assert ff == 50.0


def test_merge_speed_clusters_from_intersecting_cells():
    speed_clusters = np.full((2, 2, 4, 5, 2), np.nan)
    speed_clusters[0, 0, 0, :] = [[5.0, 10], [15.0, 20], [25.0, 30], [35.0, 40], [45.0, 5]]
    speed_clusters[0, 1, 0, :] = [[10.0, 15], [20.0, 25], [30.0, 35], [40.0, 30], [50.0, 3]]
    intersecting_cells = "[(0, 0, 0, 1.0), (0, 1, 0, 1.0)]"
    ff = merge_speed_clusters_from_intersecting_cells(speed_clusters, intersecting_cells, oneway=True, speed_limit_kph=-1, debug=True)
    assert ff == 35.0
