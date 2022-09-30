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
import logging
from pathlib import Path
from typing import Union

import h5py
import numpy as np


def load_h5_file(file_path: Union[str, Path]) -> np.ndarray:
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        return np.array(data)


def write_data_to_h5(data: np.ndarray, filename: Union[str, Path], compression="gzip", compression_level=9, dtype="uint8", verbose=False):
    with h5py.File(filename if isinstance(filename, str) else str(filename), "w", libver="latest") as f:
        if data.dtype != dtype:
            logging.warning(f"Found data with {data.dtype}, expected {dtype}.")
        if verbose:
            print(f"writing {filename} ...")
        f.create_dataset(
            # `chunks=(1, *data.shape[1:])`: optimize for row access!
            "array",
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
        )
        if verbose:
            print(f"... done writing {filename}")
