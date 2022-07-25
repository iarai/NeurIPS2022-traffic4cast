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
from pathlib import Path

import pandas
import pyarrow as pa
import pyarrow.parquet as pq

# -


def load_df_from_parquet(fn: Path):
    return pq.read_table(fn).to_pandas()


def write_df_to_parquet(df: pandas.DataFrame, fn: Path, compression="snappy"):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, fn, compression=compression)
