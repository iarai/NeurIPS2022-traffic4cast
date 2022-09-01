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

from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.prepare_training_data_eta import main


def test_prepare_training_data_eta():
    date = "1970-01-01"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        create_dummy_competition_setup(basedir=tmp_dir, city="london", train_dates=[date], skip_train_labels=True, skip_supersegments=False)
        create_dummy_competition_setup(basedir=tmp_dir, city="madrid", train_dates=[date], skip_train_labels=True, skip_supersegments=False)
        create_dummy_competition_setup(basedir=tmp_dir, city="melbourne", train_dates=[date], skip_train_labels=True, skip_supersegments=False)
        assert not (tmp_dir / "train" / "london" / "labels" / "eta_labels.parquet").exists()
        assert not (tmp_dir / "train" / "madrid" / "labels" / "eta_labels.parquet").exists()
        assert not (tmp_dir / "train" / "melbourne" / "labels" / "eta_labels.parquet").exists()
        main(["-d", str(tmp_dir)])
        assert (tmp_dir / "train" / "london" / "labels" / f"eta_labels_{date}.parquet").exists()
        assert (tmp_dir / "train" / "madrid" / "labels" / f"eta_labels_{date}.parquet").exists()
        assert (tmp_dir / "train" / "melbourne" / "labels" / f"eta_labels_{date}.parquet").exists()

        eta_df = pandas.read_parquet(tmp_dir / "train" / "london" / "labels" / f"eta_labels_{date}.parquet")
        assert len(eta_df) == 192
        assert eta_df["eta"].min() > 0
        assert eta_df["eta"].max() < 10800  # max 3h


def test_prepare_training_data_eta_outlier():
    date = "1970-01-02"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        create_dummy_competition_setup(basedir=tmp_dir, city="london", train_dates=[date], skip_train_labels=True, skip_supersegments=False)
        sc_df = pandas.read_parquet(tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet")
        sc_df.loc[((sc_df["u"] == 457298457598) & (sc_df["v"] == 9824598274857) & (sc_df["t"] == 44)), "median_speed_kph"] = 0.143
        sc_df.to_parquet(tmp_dir / "speed_classes" / "london" / f"speed_classes_{date}.parquet", compression="snappy")

        main(["-d", str(tmp_dir)])
        assert (tmp_dir / "train" / "london" / "labels" / f"eta_labels_{date}.parquet").exists()
        eta_df = pandas.read_parquet(tmp_dir / "train" / "london" / "labels" / f"eta_labels_{date}.parquet")
        assert len(eta_df) == 192

        outlier_eta_df = eta_df[(eta_df["identifier"] == "10495890,9824598274857") & (eta_df["t"] == 44)]
        print(outlier_eta_df)
        assert len(outlier_eta_df) == 1
        assert abs(outlier_eta_df["eta"].values[0] - 1880.0) < 180.0
