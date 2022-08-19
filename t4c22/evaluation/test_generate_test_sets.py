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
import json
import tempfile
from pathlib import Path

from t4c22.evaluation.generate_test_sets import BLACKLIST
from t4c22.evaluation.generate_test_sets import DATA_DAYS
from t4c22.evaluation.generate_test_sets import prepare_and_generate_test_sets
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup


def test_generate_test_sets():
    cities = ["london", "melbourne", "madrid"]
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        for city in cities:
            create_dummy_competition_setup(
                basedir=basedir,
                city=city,
                train_dates=DATA_DAYS[city]["train"],
                test_dates=DATA_DAYS[city]["test"],
                num_test_slots=num_test_slots,
                skip_golden=True,
                skip_submission=True,
                skip_tests=True,
            )
        for city in cities:
            golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
            test_input_file = basedir / "test" / city / "input" / f"counters_test.parquet"
            assert not golden_file.exists(), golden_file
            assert not test_input_file.exists(), test_input_file
        prepare_and_generate_test_sets(basedir)
        for city in cities:
            golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
            test_input_file = basedir / "test" / city / "input" / f"counters_test.parquet"
            assert golden_file.exists(), golden_file
            assert test_input_file.exists(), test_input_file

            sampled_json = basedir / "withheld" / "golden" / city / "sampled_day_t.json"
            assert sampled_json.exists(), sampled_json

            with sampled_json.open() as f:
                content = json.load(f)
                for idx, date, time_of_day in content:
                    assert date in DATA_DAYS[city]["test"] and date not in BLACKLIST, (sampled_json, idx, date, time_of_day)
