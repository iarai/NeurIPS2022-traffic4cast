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

import dp02_speed_clusters
import numpy as np
from dp02_speed_clusters import main
from h5_helpers import load_h5_file

import t4c22
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


def test_speed_clusters():
    dates = ["1970-01-01", "1970-01-02", "1970-01-03"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        dp02_speed_clusters.NUM_SLOTS = t4c22.misc.dummy_competition_setup_for_testing.NUM_SLOTS_AGGREGATED
        dp02_speed_clusters.NUM_ROWS = t4c22.misc.dummy_competition_setup_for_testing.NUM_ROWS
        dp02_speed_clusters.NUM_COLUMNS = t4c22.misc.dummy_competition_setup_for_testing.NUM_COLUMNS

        create_dummy_competition_setup(basedir=tmp_dir, city="london", skip_train_labels=True, train_dates=dates, skip_movie_15min=False)
        for date in dates:
            assert (tmp_dir / "movie_15min" / "london" / f"{date}_london_8ch_15min.h5").exists()

        speed_clusters_h5 = tmp_dir / "movie_speed_clusters" / "london" / f"speed_clusters.h5"
        assert not speed_clusters_h5.exists()

        main(["-d", str(tmp_dir), "-c", "london", "-n", "3"])

        assert speed_clusters_h5.exists()
        speed_clusters_data = load_h5_file(speed_clusters_h5)
        assert speed_clusters_data.shape == (
            t4c22.misc.dummy_competition_setup_for_testing.NUM_ROWS,
            t4c22.misc.dummy_competition_setup_for_testing.NUM_COLUMNS,
            4,
            5,
            2,
        ), speed_clusters_data.shape
        assert speed_clusters_data.dtype == np.float64, speed_clusters_data.dtype
        assert 0 <= speed_clusters_data.min() < 256, speed_clusters_data.min()
        assert 1 <= speed_clusters_data.max() < 256, speed_clusters_data.max()
        prev_cluster_speed = 0
        for cluster_speed, cluster_vol in speed_clusters_data[4, 8, 1, ...]:
            assert cluster_speed > prev_cluster_speed
            prev_cluster_speed = cluster_speed
            assert cluster_vol > 0
