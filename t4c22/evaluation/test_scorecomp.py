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
import logging
import os
import tempfile
import timeit
import zipfile
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.dataloading.test_t4c22_dataset import _inference_torch
from t4c22.evaluation.create_submission import create_submission_cc_plain_torch
from t4c22.evaluation.scorecomp import class_fractions
from t4c22.evaluation.scorecomp import evaluate_city_cc_parquet
from t4c22.evaluation.scorecomp import get_weights_from_class_fractions
from t4c22.evaluation.scorecomp import main
from t4c22.evaluation.test_create_submission import apply_model
from t4c22.evaluation.test_create_submission import DummyInfNN
from t4c22.evaluation.test_create_submission import DummyOnesNN
from t4c22.evaluation.test_create_submission import DummyRandomNN
from t4c22.evaluation.test_create_submission import DummyUniformNN
from t4c22.misc.dummy_competition_setup_for_testing import create_dummy_competition_setup
from t4c22.t4c22_config import load_cc_labels


def test_evaluate_city_cc_parquet():
    city = "london"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=False, skip_submission=False)

        submission_file = basedir / "submission" / city / "labels" / f"cc_labels_test.parquet"
        golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
        report, df = evaluate_city_cc_parquet(
            test_file=submission_file,
            golden_file=golden_file,
            city=city,
        )
        with zipfile.ZipFile(basedir / "submission" / "cc_submision.zip", "w") as z:
            z.write(submission_file, arcname=os.path.join(city, "cc_labels_test.parquet"))
        print(report)
        print(df)


def test_scorecomp_fails_with_incomplete_submission():
    city = "london"
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=False, skip_submission=False)

        submission_file = basedir / "submission" / city / "labels" / f"cc_labels_test.parquet"
        golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"

        submission_zip = basedir / "submission" / "cc_submission.zip"
        golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
        with zipfile.ZipFile(submission_zip, "w") as z:
            z.write(submission_file, arcname=os.path.join(city, "cc_labels_test.parquet"))
        with zipfile.ZipFile(golden_zip, "w") as z:
            z.write(golden_file, arcname=os.path.join(city, "cc_labels_test.parquet"))

        main(["-g", str(golden_zip), "-i", str(submission_zip)])

        print(list(basedir.rglob("**/*")))

        log_file = submission_zip.parent / "cc_submission.log"
        assert log_file.exists()
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" in content, content
        for fn in ["cc_submission.score"]:
            score_file = submission_zip.parent / fn
            assert score_file.exists()
            with score_file.open("r") as f:
                content = f.read()
                logging.info(content)
                assert content == "999", content


def test_scorecomp_inf_submission_fails(caplog):
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        submission_names = ["gogogo"]
        _create_dummy_competition_setup_with_model_class_submissions(basedir, cities, date, num_test_slots, submission_names, dummy_model_class=DummyInfNN)
        golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
        prediction_zip = basedir / "submission" / "gogogo.zip"

        caplog.set_level(logging.INFO, logger="participants-prediction")
        caplog.set_level(logging.INFO, logger="full-prediction")
        main(
            ["-g", str(golden_zip), "-i", str(prediction_zip)],
        )

        print(list(basedir.rglob("**/*")))
        log_file = prediction_zip.parent / "gogogo.log"
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" in content, content
        for file_name in ["gogogo.score"]:
            score_file = prediction_zip.parent / file_name
            assert os.path.exists(score_file), str(score_file)
            with open(score_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), 999), content


def test_scorecomp_write_log_and_score_despite_exception():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            main(["-g", os.path.join(temp_dir, "nix.zip"), "-i", os.path.join(temp_dir, "nix.zip")])
        except:  # noqa
            pass
        log_file = os.path.join(temp_dir, "nix.log")
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" in content
        for fn in ["nix.score"]:
            score_file = os.path.join(temp_dir, fn)
            assert os.path.exists(score_file)
            with open(score_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert content == "999"


@pytest.mark.parametrize(
    "args,expected_score_files,dummy_model_class",
    [
        ([], ["gogogo.score"], DummyOnesNN),
        ([], ["gogogo.score"], DummyUniformNN),
        (["--verbose"], ["gogogo.score"], DummyOnesNN),
        (["--verbose"], ["gogogo.score"], DummyUniformNN),
    ],
)
def test_scorecomp_single_submission(caplog, args, expected_score_files, dummy_model_class):
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        submission_names = ["gogogo"]
        _create_dummy_competition_setup_with_model_class_submissions(
            basedir, cities, date, num_test_slots, submission_names, dummy_model_class=dummy_model_class
        )
        golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
        prediction_zip = basedir / "submission" / "gogogo.zip"

        caplog.set_level(logging.INFO, logger="participants-prediction")
        caplog.set_level(logging.INFO, logger="full-prediction")
        main(
            ["-g", str(golden_zip), "-i", str(prediction_zip)] + args,
        )

        print(list(basedir.rglob("**/*")))
        log_file = prediction_zip.parent / "gogogo.log"
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" not in content, content
        for file_name in expected_score_files:
            score_file = prediction_zip.parent / file_name
            assert os.path.exists(score_file), str(score_file)
            with open(score_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), 1.0986122), content


def _create_dummy_competition_setup_with_model_class_submissions(
    basedir: Path, cities: List[str], date: str, num_test_slots: int, submission_names: List[str], dummy_model_class=DummyUniformNN
):
    config = {}
    for city in cities:
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[date], num_test_slots=num_test_slots, skip_golden=False, skip_submission=False)
        device = f"cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        test_dataset = T4c22Dataset(root=basedir, city=city, split="test", cachedir=Path("/tmp/processed"))

        model = dummy_model_class(num_edges=len(test_dataset.torch_road_graph_mapping.edges))

        config[city] = (test_dataset, partial(apply_model, device=device, model=model))
    for submission_name in submission_names:
        create_submission_cc_plain_torch(config=config, basedir=basedir, submission_name=submission_name)
    golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
    with zipfile.ZipFile(golden_zip, "w") as z:
        for city in cities:
            golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
            z.write(golden_file, arcname=os.path.join(city, "labels", "cc_labels_test.parquet"))


@pytest.mark.parametrize(
    "jobs,submissions,scored",
    [(1, 10, 2), (2, 10, 2), (4, 10, 2), (8, 10, 2)],
)
def test_unscored_from_folder(caplog, jobs, submissions, scored):
    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 22
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        submission_names = [f"gogogo-{i}" for i in range(submissions)]
        _create_dummy_competition_setup_with_model_class_submissions(basedir, cities, date, num_test_slots, submission_names)
        golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"

        caplog.set_level(logging.INFO, logger="participants-prediction")
        caplog.set_level(logging.INFO, logger="full-prediction")

        submissions_dir = basedir / "submission"

        # fake already scored submissions
        scored_submissions = np.random.choice(submissions, size=scored, replace=False)
        for i in scored_submissions:
            with (submissions_dir / f"{submission_names[i]}.score").open("w") as f:
                f.write("123.5")
            with (submissions_dir / f"{submission_names[i]}.log").open("w") as f:
                f.write("dummy")
            with (submissions_dir / f"{submission_names[i]}-full.log").open("w") as f:
                f.write("dummy full")

        # redirect loggers, somewhat hacky convention by name
        for i in range(submissions):
            caplog.set_level(logging.INFO, logger=f"participants-submission-{i}")
            caplog.set_level(logging.INFO)

        print("start scoring")
        scoring_time = timeit.timeit(lambda: main(["-g", str(golden_zip), "-s", str(submissions_dir), "-j", str(jobs)]), number=1)
        print(f"scoring took {scoring_time / 1000:.2f}s")

        # check the previously already scored submissions are not touched
        for i in scored_submissions:
            score_file = submissions_dir / f"{submission_names[i]}.score"
            print(score_file)
            assert score_file.exists(), score_file
            with score_file.open() as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), 123.5)
            log_file = submissions_dir / f"{submission_names[i]}.log"
            assert log_file.exists()
            with log_file.open() as f:
                content = f.read()
                logging.info(content)
                assert content == "dummy", content
            full_log_file = submissions_dir / f"{submission_names[i]}-full.log"
            assert full_log_file.exists(), full_log_file
            with full_log_file.open() as f:
                content = f.read()
                logging.info(content)
                assert "dummy full" == content, content
        #
        # check the previously unscored submissions are now scored
        unscored_submissions = [i for i in range(submissions) if i not in scored_submissions]
        for i in unscored_submissions:
            score_file = submissions_dir / f"{submission_names[i]}.score"
            print(score_file)
            assert score_file.exists(), score_file
            with score_file.open() as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), 1.0986122)
            log_file = submissions_dir / f"{submission_names[i]}.log"
            assert log_file.exists()
            with log_file.open() as f:
                content = f.read()
                logging.info(content)
                assert "completed ok" in content, content
            full_log_file = submissions_dir / f"{submission_names[i]}-full.log"
            assert full_log_file.exists(), full_log_file
            with full_log_file.open() as f:
                content = f.read()
                logging.info(content)
                assert "completed ok" in content, content


def _create_dummy_competition_setup_with_random_submissions(
    basedir: Path, cities: List[str], date: str, num_test_slots: int, submission_names: List[str], seed: int
):
    config = {}
    for city in cities:
        create_dummy_competition_setup(basedir=basedir, city=city, train_dates=[], num_test_slots=num_test_slots)

        device = f"cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        test_dataset = T4c22Dataset(root=basedir, city=city, split="test")

        model = DummyRandomNN(num_edges=len(test_dataset.torch_road_graph_mapping.edges))

        config[city] = (test_dataset, partial(apply_model, device=device, model=model))

    torch.manual_seed(seed)
    for submission_name in submission_names:
        create_submission_cc_plain_torch(config=config, basedir=basedir, submission_name=submission_name)
    golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
    with zipfile.ZipFile(golden_zip, "w") as z:
        for city in cities:
            golden_file = basedir / "withheld" / "golden" / city / "labels" / f"cc_labels_test.parquet"
            z.write(golden_file, arcname=os.path.join(city, "labels", "cc_labels_test.parquet"))

    return config


# TODO same for geometric dataset?
def test_create_submission_cc_city_plain_torch(caplog):
    """Test torch -> crossentropy is the same as torch -> pandas -> torch
    crossentropy."""

    cities = ["london", "melbourne", "madrid"]
    date = "1970-01-01"
    num_test_slots = 37
    expected_score_files = ["gogogo.score"]

    loss_before_pandas_weighted = {}
    loss_before_pandas_weighted_mask = {}
    loss_before_pandas_unweighted = {}
    loss_before_pandas_unweighted_mask = {}
    with tempfile.TemporaryDirectory() as basedir:
        basedir = Path(basedir)
        submission_names = ["gogogo"]
        # use train split to have easy access to true labels in dataset
        config = _create_dummy_competition_setup_with_random_submissions(basedir, cities, date, num_test_slots, submission_names, seed=666)

        torch.manual_seed(666)
        for city, (test_dataset, _) in config.items():
            _, y_hat = _inference_torch(ds=test_dataset, model=DummyRandomNN(num_edges=len(test_dataset.torch_road_graph_mapping.edges)))

            df_y = load_cc_labels(basedir, city=city, split="test", df_filter=None)
            y = torch.cat([test_dataset.torch_road_graph_mapping._df_cc_to_torch(df_y[df_y["test_idx"] == test_idx]) for test_idx in range(num_test_slots)])

            class_weights = get_weights_from_class_fractions([class_fractions[city][c] for c in ["green", "yellow", "red"]])
            loss_f_weighted = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float(), ignore_index=-1)

            loss_f_unweighted = torch.nn.CrossEntropyLoss(ignore_index=-1)

            y = y.nan_to_num(-1).long()

            loss_before_pandas_weighted[city] = loss_f_weighted(y_hat, y)
            loss_before_pandas_unweighted[city] = loss_f_unweighted(y_hat, y)

            y_hat = y_hat[y >= 0]
            y = y[y >= 0]
            loss_before_pandas_weighted_mask[city] = loss_f_weighted(y_hat, y)
            loss_before_pandas_unweighted_mask[city] = loss_f_unweighted(y_hat, y)

            assert np.isclose(loss_before_pandas_weighted[city], loss_before_pandas_weighted_mask[city])
            assert np.isclose(loss_before_pandas_unweighted[city], loss_before_pandas_unweighted_mask[city])

        golden_zip = basedir / "withheld" / "golden" / "cc_golden.zip"
        prediction_zip = basedir / "submission" / "gogogo.zip"

        caplog.set_level(logging.INFO, logger="participants-prediction")
        caplog.set_level(logging.INFO, logger="full-prediction")
        main(
            ["-g", str(golden_zip), "-i", str(prediction_zip)],
        )

        print(list(basedir.rglob("**/*")))
        log_file = prediction_zip.parent / "gogogo-full.log"
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            print(content)
        log_file = prediction_zip.parent / "gogogo.log"
        with open(log_file, "r") as f:
            content = f.read()
            logging.info(content)
            assert "contact us for details" not in content, content
        with (prediction_zip.parent / "gogogo.score.json").open() as f:
            content = json.load(f)
            print(content)
            for city in cities:
                assert np.isclose(loss_before_pandas_weighted[city], content[city]["all_weighted"]["loss"])
                assert np.isclose(loss_before_pandas_unweighted[city], content[city]["all_unweighted"]["loss"])

        for file_name in expected_score_files:
            score_file = prediction_zip.parent / file_name
            assert os.path.exists(score_file), str(score_file)
            with open(score_file, "r") as f:
                content = f.read()
                logging.info(content)
                assert np.isclose(float(content), np.mean([loss_before_pandas_weighted[city] for city in cities])), content
