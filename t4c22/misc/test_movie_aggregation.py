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

import t4c22
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.misc.movie_aggregation import load_h5_file
from t4c22.misc.movie_aggregation import main

# -


def test_movie_aggregation():
    date = "1970-01-01"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        t4c22.misc.movie_aggregation.NUM_SLOTS_NON_AGGREGATED = t4c22.misc.dummy_competition_setup_for_testing.NUM_SLOTS_NON_AGGREGATED
        t4c22.misc.movie_aggregation.NUM_SLOTS_AGGREGATED = t4c22.misc.dummy_competition_setup_for_testing.NUM_SLOTS_AGGREGATED
        t4c22.misc.movie_aggregation.NUM_ROWS = t4c22.misc.dummy_competition_setup_for_testing.NUM_ROWS
        t4c22.misc.movie_aggregation.NUM_COLUMNS = t4c22.misc.dummy_competition_setup_for_testing.NUM_COLUMNS

        create_dummy_competition_setup(basedir=tmp_dir, city="london", skip_train_labels=True, train_dates=[date], skip_movie=False)
        assert (tmp_dir / "movie" / "london" / f"{date}_london_8ch.h5").exists()

        london_8ch_15_h5 = tmp_dir / "movie_15min" / "london" / f"{date}_london_8ch_15min.h5"
        assert not london_8ch_15_h5.exists()

        main(["-d", str(tmp_dir)])

        assert london_8ch_15_h5.exists()
        london_8ch_15_data = load_h5_file(london_8ch_15_h5)
        assert london_8ch_15_data.shape == (
            t4c22.misc.dummy_competition_setup_for_testing.NUM_SLOTS_AGGREGATED,
            t4c22.misc.dummy_competition_setup_for_testing.NUM_ROWS,
            t4c22.misc.dummy_competition_setup_for_testing.NUM_COLUMNS,
            8,
        ), london_8ch_15_data.shape
        assert london_8ch_15_data.dtype == np.float64, london_8ch_15_data.dtype
