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
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import tqdm

from t4c22.dataloading.t4c22_dataset import T4c22Competitions
from t4c22.evaluation.generate_test_sets import CITIES
from t4c22.misc.parquet_helpers import load_df_from_parquet
from t4c22.t4c22_config import NUM_SUPERSEGMENTS

EXPECTED_NUM_TRAINING_LABEL_FILES = {"london": 110, "madrid": 109, "melbourne": 108}

# these counts are for the release published 2022-09-02 with 4012 supersegments for london, ...
CC_SUMS = {"london": 516697931.0, "madrid": 765837597.0, "melbourne": 132172557.0}
ETA_SUMS = {"london": 14940784306.865263, "madrid": 13610842992.839167, "melbourne": 7184138390.886182}


def sanity_check_labels(data_folder: Path, competitions: Optional[List[T4c22Competitions]]):
    summary = []
    if competitions is None or T4c22Competitions.CORE in competitions:
        print(f"/ start core competition check")
        d = {}
        for city in CITIES:
            cc_sum = 0.0
            label_file_list = list((data_folder / "train" / city).rglob("cc_labels*.parquet"))
            assert len(label_file_list) == EXPECTED_NUM_TRAINING_LABEL_FILES[city], (len(label_file_list), EXPECTED_NUM_TRAINING_LABEL_FILES[city])
            for file in tqdm.tqdm(label_file_list, desc=city):
                df = load_df_from_parquet(file)
                assert len(df[df["cc"].isna()]) == 0, (file, df["c"].isna())
                assert df["cc"].min() >= 0, (file, df["cc"].min())
                assert df["cc"].max() <= 3, (file, df["cc"].max())
                cc_sum += df["cc"].astype(np.float64).sum()
            d[city] = cc_sum
            cc_sum_check = np.isclose(cc_sum, CC_SUMS[city])
            python_check = "(\u2713)" if cc_sum_check else "(\u2717)"
            msg = str((city, (cc_sum, CC_SUMS[city]))) if not cc_sum_check else city
            msg = f"{python_check} cc sum check {msg}"
            print("")
            print(msg)
            summary.append(msg)

        print(json.dumps(d))
        print(f"\\ end core competition check -> OK!")

    if competitions is None or T4c22Competitions.EXTENDED in competitions:
        print(f"/ start extended competition check")
        d = {}
        for city in CITIES:
            label_file_list = list((data_folder / "train" / city).rglob("eta_labels*.parquet"))
            eta_sum = 0.0
            assert len(label_file_list) == EXPECTED_NUM_TRAINING_LABEL_FILES[city], (len(label_file_list), EXPECTED_NUM_TRAINING_LABEL_FILES[city])
            for file in tqdm.tqdm(label_file_list, desc=city):
                df = load_df_from_parquet(file)
                assert len(df[df["eta"].isna()]) == 0, df["eta"].isna()
                num_supersegments_city = NUM_SUPERSEGMENTS[city]
                assert len(df) == num_supersegments_city * 96, (len(df), num_supersegments_city * 96)

                assert (df.groupby(["day", "t"], sort=False).agg(lambda x: len(x.unique()))["identifier"] == num_supersegments_city).all()

                eta_sum += df["eta"].astype(np.float64).sum()
            d[city] = eta_sum
            eta_sum_check = np.isclose(eta_sum, ETA_SUMS[city])
            python_check = "(\u2713)" if eta_sum_check else "(\u2717)"
            msg = str((city, (eta_sum, ETA_SUMS[city]))) if not eta_sum_check else city
            msg = f"{python_check} eta sum check {msg}"
            print("")
            print(msg)
            print(msg)
            summary.append(msg)

        print("")
        print(json.dumps(d))
        print(f"\\ end extended competition check -> OK!")

    print("")
    print("")
    print("Summary:")
    print("\n".join(summary))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("This script takes the T4c22 data directory and checks the training label files."))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data",
        required=True,
    )
    parser.add_argument("-c", "--competition", type=T4c22Competitions, help="Competition", required=False, default=None, nargs="+")
    return parser


def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(argv)
        params = vars(params)
        data_folder = Path(params["data_folder"])
        competitions = params["competition"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        raise e
    sanity_check_labels(data_folder=data_folder, competitions=competitions)


if __name__ == "__main__":
    main(sys.argv[1:])
