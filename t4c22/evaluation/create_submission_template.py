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
import datetime
from functools import partial
from pathlib import Path

import torch

import t4c22
from t4c22.dataloading.t4c22_dataset import T4c22Dataset  # noqa
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.evaluation.create_submission import create_submission_cc_plain_torch
from t4c22.evaluation.create_submission import create_submission_cc_torch_geometric
from t4c22.evaluation.test_create_submission import apply_model
from t4c22.evaluation.test_create_submission import apply_model_geometric
from t4c22.evaluation.test_create_submission import DummyRandomNN
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import load_basedir

# The submission zip for the core competition must have the following file structure:
#     ```
# london/labels/cc_labels_test.parquet
# madrid/labels/cc_labels_test.parquet
# melbourne/labels/cc_labels_test.parquet
# ```


def main(basedir: Path, submission_name: str, model_class, dataset_class, geom=False):
    t4c_apply_basic_logging_config(loglevel="DEBUG")

    cities = ["london", "melbourne", "madrid"]

    config = {}
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    for city in cities:
        test_dataset = dataset_class(root=basedir, city=city, split="test")

        model = model_class(num_edges=len(test_dataset.torch_road_graph_mapping.edges))
        # we go without checkpoint loading in unit test, add for your model:
        # load.load_state_dict(torch.load(your-checkpoint, map_location=device)) #noqa

        if not geom:
            config[city] = (test_dataset, partial(apply_model, device=device, model=model))
        else:
            config[city] = (test_dataset, partial(apply_model_geometric, device=device, model=model))

    if geom:
        create_submission_cc_torch_geometric(config=config, basedir=basedir, submission_name=submission_name)
    else:
        create_submission_cc_plain_torch(config=config, basedir=basedir, submission_name=submission_name)


if __name__ == "__main__":
    # model_class = DummyUniformNN #noqa
    # model_class = DummyRandomNN#noqa
    # dataset_class = T4c22Dataset#noqa
    # geom = False#noqa

    model_class = DummyRandomNN
    dataset_class = T4c22GeometricDataset
    geom = True

    submission_name = f"{model_class.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(submission_name)
    basedir = load_basedir(fn="t4c22_config.json", pkg=t4c22)

    main(basedir=basedir, submission_name=submission_name, model_class=model_class, dataset_class=dataset_class, geom=geom)
