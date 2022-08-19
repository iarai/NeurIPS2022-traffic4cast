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
import os
import zipfile
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import torch
import torch_geometric
import tqdm
from torch import Tensor

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.misc.parquet_helpers import write_df_to_parquet

CITIES = ["london", "melbourne", "madrid"]


@torch.no_grad()
def inference_cc_city_torch_geometric_to_pandas(test_dataset: T4c22GeometricDataset, predict: Callable[[torch_geometric.data], torch.Tensor]) -> pd.DataFrame:
    dfs = []
    for idx, data in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
        y_hat: Tensor = predict(data)
        df = test_dataset.torch_road_graph_mapping._torch_to_df_cc(data=y_hat, day="test", t=idx)
        dfs.append(df)
    df = pd.concat(dfs)
    df["test_idx"] = df["t"]
    del df["day"]
    del df["t"]
    return df


@torch.no_grad()
def inference_cc_city_plain_torch_to_pandas(test_dataset: T4c22Dataset, predict: Callable[[Any], torch.Tensor]) -> pd.DataFrame:
    dfs = []
    for idx, data in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
        y_hat: Tensor = predict(data)
        df = test_dataset.torch_road_graph_mapping._torch_to_df_cc(data=y_hat, day="test", t=idx)
        dfs.append(df)
    df = pd.concat(dfs)
    df["test_idx"] = df["t"]
    del df["day"]
    del df["t"]
    return df


def create_submission_cc_torch_geometric(
    config: Dict[str, Tuple[T4c22GeometricDataset, Callable[[torch_geometric.data], torch.Tensor]]],
    submission_name: str,
    basedir: Path,
    cities: Optional[List[str]] = None,
):
    if cities is None:
        cities = CITIES
    for city in cities:
        test_dataset, predict = config[city]
        df_city = inference_cc_city_torch_geometric_to_pandas(test_dataset=test_dataset, predict=predict)
        (basedir / "submission" / submission_name / city / "labels").mkdir(exist_ok=True, parents=True)
        write_df_to_parquet(df=df_city, fn=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet")

    submission_zip = basedir / "submission" / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            z.write(
                filename=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet",
                arcname=os.path.join(city, "labels", f"cc_labels_test.parquet"),
            )
    print(submission_zip)


def create_submission_cc_plain_torch(
    config: Dict[str, Tuple[T4c22Dataset, Callable[[Any], torch.Tensor]]], submission_name: str, basedir: Path, cities: Optional[List[str]] = None
):
    if cities is None:
        cities = CITIES

    for city in cities:
        test_dataset, predict = config[city]
        df_city = inference_cc_city_plain_torch_to_pandas(test_dataset=test_dataset, predict=predict)
        (basedir / "submission" / submission_name / city / "labels").mkdir(exist_ok=True, parents=True)
        write_df_to_parquet(df=df_city, fn=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet")

    submission_zip = basedir / "submission" / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            z.write(
                filename=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet",
                arcname=os.path.join(city, "labels", f"cc_labels_test.parquet"),
            )
    print(submission_zip)
