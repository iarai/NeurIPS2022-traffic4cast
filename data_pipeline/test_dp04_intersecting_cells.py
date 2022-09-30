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
from ast import literal_eval
from pathlib import Path

import data_helpers as dp_data_helpers
import pandas
from dp04_intersecting_cells import main
from test_data_helpers import assert_intersecting_cells_equal

from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


def test_intersecting_cells():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        dp_data_helpers.T4C_BBOXES = {
            "london": {"bounds": [5167600, 5170000, -369000, -367000]},
        }
        dp_data_helpers.NUM_ROWS = 24
        dp_data_helpers.NUM_COLUMNS = 20

        create_dummy_competition_setup(basedir=tmp_dir, city="london", train_dates=[], skip_train_labels=True)
        assert (tmp_dir / "road_graph" / "london" / "road_graph_edges.parquet").exists()

        cells_pq = tmp_dir / "road_graph" / "london" / "road_graph_intersecting_cells.parquet"
        assert not cells_pq.exists()

        main(["-d", str(tmp_dir), "-c", "london"])

        assert cells_pq.exists()
        cells_df = pandas.read_parquet(cells_pq)
        assert len(cells_df) == 6

        cells = cells_df[(cells_df["u"] == 10495890) & (cells_df["v"] == 457298457598)]
        cells = literal_eval(cells["intersecting_cells"].values[0])
        assert len(cells) == 6
        expected_cells = [(2, 2, 3, 0.25), (2, 3, 3, 0.25), (1, 6, 3, 0.0), (3, 1, 3, 0.0), (2, 4, 3, 0.25), (2, 5, 3, 0.25)]
        assert_intersecting_cells_equal(cells, expected_cells)

        # Re-run to check that the file doesn't get accidentally overwritten
        previous_mtime = cells_pq.stat().st_mtime
        main(["-d", str(tmp_dir), "-c", "london"])
        assert cells_pq.stat().st_mtime == previous_mtime

        # Re-run and force overwriting of the cell mappings
        previous_mtime = cells_pq.stat().st_mtime
        main(["-d", str(tmp_dir), "-c", "london", "-f"])
        assert cells_pq.stat().st_mtime > previous_mtime
