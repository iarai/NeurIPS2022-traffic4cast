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

import pandas
from dp06_speed_classes import main

from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


def test_speed_classes():
    dates = ["1970-01-01", "1970-01-02"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        create_dummy_competition_setup(
            basedir=tmp_dir, city="london", skip_train_labels=True, skip_speed_classes=True, train_dates=dates, skip_movie_15min=False, skip_free_flow=False
        )
        assert (tmp_dir / "road_graph" / "london" / "road_graph_freeflow.parquet").exists()
        for date in dates:
            assert (tmp_dir / "movie_15min" / "london" / f"{date}_london_8ch_15min.h5").exists()
            assert not (tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet").exists()

        main(["-d", str(tmp_dir), "-c", "london"])

        for date in dates:
            assert (tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet").exists()

        sc_df = pandas.read_parquet(tmp_dir / "speed_classes" / "london" / f"speed_classes_{dates[0]}.parquet")
        assert len(sc_df) == 24
        assert sc_df["median_speed_kph"].min() > 0
        assert sc_df["median_speed_kph"].max() < 120
