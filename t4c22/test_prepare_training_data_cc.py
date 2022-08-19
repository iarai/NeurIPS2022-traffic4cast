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

from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.prepare_training_data_cc import main


def test_prepare_training_data_cc():
    date = "1970-01-01"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        create_dummy_competition_setup(
            basedir=tmp_dir,
            city="london",
            skip_train_labels=True,
            train_dates=[date],
        )
        create_dummy_competition_setup(
            basedir=tmp_dir,
            city="madrid",
            skip_train_labels=True,
            train_dates=[date],
        )
        create_dummy_competition_setup(basedir=tmp_dir, city="melbourne", skip_train_labels=True, train_dates=[date])
        assert not (tmp_dir / "train" / "london" / "labels" / "cc_labels.parquet").exists()
        assert not (tmp_dir / "train" / "madrid" / "labels" / "cc_labels.parquet").exists()
        assert not (tmp_dir / "train" / "melbourne" / "labels" / "cc_labels.parquet").exists()
        main(["-d", str(tmp_dir)])
        assert (tmp_dir / "train" / "london" / "labels" / f"cc_labels_{date}.parquet").exists()
        assert (tmp_dir / "train" / "madrid" / "labels" / f"cc_labels_{date}.parquet").exists()
        assert (tmp_dir / "train" / "melbourne" / "labels" / f"cc_labels_{date}.parquet").exists()
