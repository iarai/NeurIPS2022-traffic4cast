#!/usr/bin/python3
# -*- coding: utf-8 -*-
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
#
#   http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import argparse
import datetime
import glob
import json
import logging
import os
import sys
import tempfile
import time
import zipfile
from enum import Enum
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import tqdm


# from t4c22.dataloading.t4c22_dataset import T4c22Competitions # noqa
class T4c22Competitions(Enum):
    CORE = "cc"
    EXTENDED = "eta"


# from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions # noqa
def get_weights_from_class_fractions(class_fractions):
    n = np.sum(class_fractions)
    return [n / (c * 3) for c in class_fractions]


# from t4c22.misc.parquet_helpers import load_df_from_parquet # noqa
def load_df_from_parquet(fn: Path):
    return pq.read_table(fn).to_pandas()


# from t4c22.t4c22_config import class_fractions # noqa
class_fractions = {
    "london": ({"green": 0.5367906303432076, "yellow": 0.35138063340805714, "red": 0.11182873624873524}),
    "madrid": {"green": 0.4976221039083026, "yellow": 0.3829591430424158, "red": 0.1194187530492816},
    "melbourne": {"green": 0.7018930324884697, "yellow": 0.2223245729555099, "red": 0.0757823945560204},
}

MAXSIZE = 800 * 1024 * 1024 * 8
dataset_filters = {
    "all": lambda df: df,
    # "counter_distance_zero": lambda df: df[df["counter_distance"] == 0],#noqa
    # "workdays_daytime": lambda df: df[(df["weekday"]<5)&(df["h"]>7)&(df["h"]<22)], #noqa
}
SCOREFILE_CONFIG = {
    T4c22Competitions.CORE: {
        ".score": "all_weighted",
    },
    T4c22Competitions.EXTENDED: {
        ".score": "all",
    },
}
SCORES_CONFIG = {T4c22Competitions.CORE: ["all_weighted", "all_unweighted"], T4c22Competitions.EXTENDED: ["all"]}


EXPECTED_NUM_SLOTS = 100
EXPECTED_NUM_ITEMS = {
    T4c22Competitions.CORE.value: {
        "london": 132414 * EXPECTED_NUM_SLOTS,
        "madrid": 121902 * EXPECTED_NUM_SLOTS,
        "melbourne": 94871 * EXPECTED_NUM_SLOTS,
    },
    T4c22Competitions.EXTENDED.value: {
        "london": 4012 * EXPECTED_NUM_SLOTS,
        "madrid": 3969 * EXPECTED_NUM_SLOTS,
        "melbourne": 3246 * EXPECTED_NUM_SLOTS,
    },
}


def _merge_pred_true_cc(df_pred, df_true):
    logging.debug(f"evaluate_submission_cc {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pred_columns = ["u", "v", "test_idx", "logit_green", "logit_yellow", "logit_red"]
    for k in pred_columns:
        assert k in df_pred.columns, (k, df_pred.columns)
    # avoid clashes
    df_pred = df_pred[pred_columns]

    true_columns = ["u", "v", "test_idx", "cc"]
    for k in true_columns:
        assert k in df_true.columns, (k, df_true.columns)
    df_true = df_true[true_columns]

    assert df_true["cc"].min() >= 0
    assert df_true["cc"].max() <= 3

    logging.debug(f"start merge {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_merged = df_pred.merge(df_true, left_on=["u", "v", "test_idx"], right_on=["u", "v", "test_idx"], suffixes=["_pred", ""])
    assert len(df_merged) == len(df_true), (len(df_merged), len(df_true))
    logging.debug(f"end merge {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df_merged


def _merge_pred_true_eta(df_pred, df_true):
    logging.debug(f"evaluate_submission_cc {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pred_columns = ["identifier", "test_idx", "eta"]
    for k in pred_columns:
        assert k in df_pred.columns, (k, df_pred.columns)
    # avoid clashes
    df_pred = df_pred[pred_columns]

    true_columns = ["identifier", "test_idx", "eta"]
    for k in true_columns:
        assert k in df_true.columns, (k, df_true.columns)
    df_true = df_true[true_columns]

    logging.debug(f"start merge {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_merged = df_pred.merge(df_true, left_on=["identifier", "test_idx"], right_on=["identifier", "test_idx"], suffixes=["_pred", ""])
    assert len(df_merged) == len(df_true), (len(df_merged), len(df_true))
    logging.debug(f"end merge {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df_merged


def evaluate_city_cc(df_pred, df_true, class_fractions: Dict[str, float], df_merged=None) -> Tuple[Dict, pd.DataFrame]:
    if df_merged is None:
        df_merged = _merge_pred_true_cc(df_pred, df_true)
    report = {}

    for key, filter_func in tqdm.tqdm(dataset_filters.items()):
        df = filter_func(df_merged)

        y_hat = torch.tensor(df[["logit_green", "logit_yellow", "logit_red"]].to_numpy()).float()

        y = torch.tensor(df["cc"].to_numpy()).long() - 1
        assert torch.min(y) >= -1
        assert torch.max(y) <= 2

        class_weights = get_weights_from_class_fractions([class_fractions[c] for c in ["green", "yellow", "red"]])
        loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float(), ignore_index=-1)
        loss = float(loss_f(y_hat, y).cpu().numpy())
        report[key + "_weighted"] = {"metric": "torch.nn.CrossEntropyLoss", "loss": loss, "class_weights": class_weights, "class_fractions": class_fractions}

        loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = float(loss_f(y_hat, y).cpu().numpy())
        report[key + "_unweighted"] = {"metric": "torch.nn.CrossEntropyLoss", "loss": loss, "class_weights": None, "class_fractions": None}

    return report, df_merged


def evaluate_city_eta(df_pred, df_true, df_merged=None) -> Tuple[Dict, pd.DataFrame]:
    df_merged = _merge_pred_true_eta(df_pred, df_true)
    loss_f = torch.nn.L1Loss()
    y_hat = torch.tensor(df_merged["eta_pred"].to_numpy()).float()
    y = torch.tensor(df_merged["eta"].to_numpy()).float()
    loss = float(loss_f(y_hat, y).numpy())
    return {
        "all": {
            "metric": "torch.nn.L1Loss",
            "loss": loss,
        }
    }, df_merged


def evaluate_city_cc_parquet(test_file: str, golden_file: str, city: str, participants_logger_name: str):
    df_pred = load_df_from_parquet(test_file)

    expected_num_items = EXPECTED_NUM_ITEMS[T4c22Competitions.CORE.value][city]
    if len(df_pred) != expected_num_items:
        msg = f"Your submission for core competition (cc) for {city} is expected to have length {expected_num_items}, found {len(df_pred)}. "
        participants_logger = logging.getLogger(participants_logger_name)
        participants_logger.error(msg)
        raise Exception(msg)
    df_true = load_df_from_parquet(golden_file)
    return evaluate_city_cc(df_pred, df_true, class_fractions=class_fractions[city])


def evaluate_city_eta_parquet(test_file: str, golden_file: str, city: str, participants_logger_name: str):
    df_pred = load_df_from_parquet(test_file)
    expected_num_items = EXPECTED_NUM_ITEMS[T4c22Competitions.EXTENDED.value][city]
    if len(df_pred) != expected_num_items:
        msg = f"Your submission for extended competition (eta) for {city} is expected to have length {expected_num_items}, found {len(df_pred)}. "
        participants_logger = logging.getLogger(participants_logger_name)
        participants_logger.error(msg)
        raise Exception(msg)
    df_true = load_df_from_parquet(golden_file)
    return evaluate_city_eta(df_pred, df_true)


def sanitize(a):
    if hasattr(a, "shape"):
        return a.tolist()
    return a


def average_city_scores(scores_dict: Dict, city_keys: List[str], cities: List[str]) -> Dict:
    scores_dict["all"] = {}
    for k in city_keys:
        scores_dict["all"][k] = np.sum([scores_dict[city][k]["loss"] for city in cities]) / len(cities)
    for _, d in scores_dict.items():
        for ki, v in d.items():
            d[ki] = sanitize(v)


def do_score(ground_truth_archive: str, input_archive: str, participants_logger_name, competition: T4c22Competitions) -> Tuple[float, Dict]:
    start_time = time.time()
    participants_logger = logging.getLogger(participants_logger_name)

    archive_size = os.path.getsize(input_archive)
    participants_logger.info(f"{os.path.basename(input_archive)} has size {archive_size / (1024 * 1024):.3f}MB")
    if archive_size > MAXSIZE:
        msg = (
            f"Your submission archive is too large (> {MAXSIZE / (1024 * 1024):.2f}MB). "
            f"Have you activated HDF5 compression? Please adapt your files as necessary and resubmit."
        )
        participants_logger.error(msg)
        raise Exception(msg)
    with zipfile.ZipFile(input_archive) as prediction_f:
        prediction_file_list = [f for f in prediction_f.namelist() if "test" in f and f.endswith(".parquet")]
    with zipfile.ZipFile(ground_truth_archive) as ground_truth_f:
        ground_truth_archive_list = [f for f in ground_truth_f.namelist() if "test" in f and f.endswith(".parquet")]
    if set(prediction_file_list) != set(ground_truth_archive_list):
        msg = (
            f"Your submission differs from the ground truth file list. Please adapt the submitted archive as necessary and resubmit. "
            f"Missing files: {set(ground_truth_archive_list).difference(prediction_file_list)}. "
            f"Unexpected files: {set(prediction_file_list).difference(ground_truth_archive_list)}."
        )
        participants_logger.error(msg)
        raise Exception(msg)

    scores_dict = {}

    with tempfile.TemporaryDirectory() as temp_dir_prediction:
        with tempfile.TemporaryDirectory() as temp_dir_ground_truth:
            with zipfile.ZipFile(ground_truth_archive) as ground_truth_f:
                with zipfile.ZipFile(input_archive) as prediction_f:
                    for f in ground_truth_archive_list:
                        city_name = f.split("/")[0]
                        prediction_f_extracted = prediction_f.extract(f, path=temp_dir_prediction)
                        ground_truth_f_extracted = ground_truth_f.extract(f, path=temp_dir_ground_truth)
                        if competition == T4c22Competitions.CORE:
                            report, _ = evaluate_city_cc_parquet(
                                test_file=prediction_f_extracted,
                                golden_file=ground_truth_f_extracted,
                                city=city_name,
                                participants_logger_name=participants_logger_name,
                            )
                        else:
                            report, _ = evaluate_city_eta_parquet(
                                test_file=prediction_f_extracted,
                                golden_file=ground_truth_f_extracted,
                                city=city_name,
                                participants_logger_name=participants_logger_name,
                            )

                        logging.info(f"City scores {city_name}")
                        scores_dict[city_name] = report

    average_city_scores(scores_dict, city_keys=SCORES_CONFIG[competition], cities=["london", "madrid", "melbourne"])
    score = scores_dict["all"][SCOREFILE_CONFIG[competition][".score"]]
    elapsed_seconds = time.time() - start_time
    logging.info(f"scoring {os.path.basename(input_archive)} took {elapsed_seconds :.1f}s")
    logging.info(f"Scores {scores_dict}")
    return score, scores_dict


def score_participant(input_archive: str, ground_truth_archive: str, competition: T4c22Competitions):
    submission_id = os.path.basename(input_archive).replace(".zip", "")

    full_handler = logging.FileHandler(input_archive.replace(".zip", "-full.log"))
    json_score_file = input_archive.replace(".zip", ".score.json")
    full_handler.setLevel(logging.INFO)
    full_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
    full_logger = logging.getLogger()
    full_logger.addHandler(full_handler)

    # create the log and score files with bad score in case an exception happens
    participants_handler = logging.FileHandler(input_archive.replace(".zip", ".log"))
    participants_handler.setLevel(logging.INFO)
    participants_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
    participants_logger_name = f"participants-{submission_id}"
    participants_logger = logging.getLogger(participants_logger_name)
    participants_logger.addHandler(participants_handler)
    input_archive_basename = os.path.basename(input_archive)
    participants_logger.info(f"start scoring of {input_archive_basename}")
    participants_logger.addHandler(full_handler)

    score_file_extensions = SCOREFILE_CONFIG[competition]
    for score_file_ext in score_file_extensions:
        score_file = input_archive.replace(".zip", score_file_ext)
        with open(score_file, "w") as f:
            f.write("999")
    try:
        # do scoring and update score file
        vanilla_score, scores_dict = do_score(
            input_archive=input_archive, ground_truth_archive=ground_truth_archive, participants_logger_name=participants_logger_name, competition=competition
        )
        with open(json_score_file, "w") as f:
            json.dump(scores_dict, f)

        for score_file_ext, score_key in score_file_extensions.items():
            score_file = input_archive.replace(".zip", score_file_ext)
            score = scores_dict["all"][score_key]
            # error code in leaderboard must be 999 (if not parseable as float, leaderboard will display unscored), so fail before writing nan/inf to the score file.
            try:
                assert not np.isnan(score), score
                assert not np.isinf(score), score
                with open(score_file, "w") as f:
                    f.write(str(score))
            except AssertionError as e:
                participants_logger.error(f"Evaluation returned '{score}'")
                raise e
            participants_logger.info(f"Evaluation completed ok with score {vanilla_score} for {input_archive_basename}")
        participants_handler.flush()

    except Exception as e:
        logging.exception(f"There was an error during execution, please review", exc_info=e)
        participants_logger.error(f"Evaluation errors for {input_archive_basename}, contact us for details via github issues.")


def score_unscored_participants(ground_truth_archive, jobs, submissions_folder, competition: T4c22Competitions):
    all_submissions = [z.replace(".zip", "") for z in glob.glob(f"{submissions_folder}/*.zip")]
    unscored = [s for s in all_submissions if not os.path.exists(os.path.join(submissions_folder, f"{s}.score"))]
    unscored_zips = [os.path.join(submissions_folder, f"{s}.zip") for s in unscored]
    if jobs == 0:
        for u in unscored_zips:
            score_participant(u, ground_truth_archive=ground_truth_archive, competition=competition)
    else:
        with Pool(processes=jobs) as pool:
            _ = list(pool.imap_unordered(partial(score_participant, ground_truth_archive=ground_truth_archive, competition=competition), unscored_zips))


def create_parser() -> argparse.ArgumentParser:
    """Create test files and copy static and dynamic h5 files to the same place
    and tar them."""
    parser = argparse.ArgumentParser(
        description=(
            "This script takes either the path for an individual T4c 2021 submission zip file and evaluates the total "
            "score or it scans through the submission directory to compute scores for all files missing a score."
        )
    )
    # data arguments
    parser.add_argument(
        "-g",
        "--ground_truth_archive",
        type=str,
        help="zip file containing the ground truth",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input_archive",
        type=str,
        help="single participant submission zip archive",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--submissions_folder",
        type=str,
        help="folder containing participant submissions",
        required=False,
    )
    parser.add_argument("-j", "--jobs", type=int, help="Number of jobs to run in parallel", required=False, default=1)
    parser.add_argument("-v", "--verbose", help="Do not silence caught exceptions.", required=False, default=False, action="store_true")

    parser.add_argument("-c", "--competition", type=T4c22Competitions, help="Competition", required=False, default=T4c22Competitions.CORE)

    return parser


def main(args):  # noqa C901
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"), format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(args)
        params = vars(params)
        ground_truth_archive = params["ground_truth_archive"]
        jobs = params["jobs"]
        verbose = params["verbose"]
        competition = params["competition"]

        if params["input_archive"] is not None:
            try:
                score_participant(input_archive=params["input_archive"], ground_truth_archive=ground_truth_archive, competition=competition)
            except Exception as e:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                if verbose:
                    raise e
        elif params["submissions_folder"] is not None:
            try:
                score_unscored_participants(
                    ground_truth_archive=ground_truth_archive, jobs=jobs, submissions_folder=params["submissions_folder"], competition=competition
                )
            except Exception as e:
                # exceptions are logged to participants and full log. Should we remove score file in case of runtime exception (OOM)?
                if verbose:
                    raise e
        else:
            raise Exception("Either input archive or submissions folder must be given")
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
