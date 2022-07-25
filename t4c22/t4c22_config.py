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
import importlib.resources as pkg_resources
import json
import re
from functools import partial
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import pyarrow.parquet as pq

import t4c22

# -


# -----------------------------------------------------------------------------------------------------
# data base directore from config file
# -----------------------------------------------------------------------------------------------------
def load_basedir(fn: Path = None, pkg=t4c22) -> Path:
    """Load t4c22 data basedir from central config file.

    Parameters
    ----------
    fn: json file with BASEDIR entry; defaults to `Path("t4c22_config.json")`
    pkg: if to load from resource path; defaults to `t4c22`

    Returns
    -------
    """
    if fn is None:
        fn = Path("t4c22_config.json")
    if pkg is None:
        with fn.open() as f:
            config = json.load(f)
            return Path(config["BASEDIR"])
    else:
        with pkg_resources.path(pkg, fn) as p:
            with open(p) as f:
                config = json.load(f)
                return Path(config["BASEDIR"])


# -----------------------------------------------------------------------------------------------------
# FILTERING DATA
# -----------------------------------------------------------------------------------------------------


def day_t_filter(day: str, t: int, day_whitelist=None, t_whitelist=None, weekday_whitelist=None) -> bool:
    """Filter for day and t."""
    if day_whitelist is not None:
        return day in day_whitelist
    if t_whitelist is not None:
        return t in t_whitelist
    if weekday_whitelist is not None:
        weekday = datetime.datetime.strptime(day, "%Y-%m-%d").weekday()
        return weekday in weekday_whitelist
    return True


DAY_T_FILTER = Callable[[str, int], bool]
DF_FILTER = Callable[[pd.DataFrame], pd.DataFrame]

day_t_filter_weekdays_daytime_only: DAY_T_FILTER = partial(day_t_filter, t_whitelist=set(range(6 * 4, 22 * 4)), weekday_whitelist=set(range(6)))


def day_t_filter_to_df_filter(df: pd.DataFrame, filter: DAY_T_FILTER, tmp_column_name="_included") -> pd.DataFrame:
    """Filter frame on day and t columns through given filter.

    Parameters
    ----------
    df
    filter: filter taking day and t
    tmp_column_name

    Returns
    -------
    """
    assert tmp_column_name not in df.columns
    df[tmp_column_name] = [filter(day, t) for day, t in zip(df["day"], df["t"])]
    df = df[df[tmp_column_name]]
    del df[tmp_column_name]
    return df


df_filter_weekdays_daytime_only: DF_FILTER = partial(day_t_filter_to_df_filter, filter=day_t_filter_weekdays_daytime_only)


# -----------------------------------------------------------------------------------------------------
# HELPERS FOR LOADING COMPETITION DATA (MOSTLY FROM PARQUET)
# -----------------------------------------------------------------------------------------------------


def load_road_graph(basedir: Path, city: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper for loading edges and nodes data frame from basedir for the given
    city."""
    fn = basedir / "road_graph" / city / "road_graph_edges.parquet"
    df_edges = pq.read_table(fn).to_pandas()
    fn = basedir / "road_graph" / city / "road_graph_nodes.parquet"
    df_nodes = pq.read_table(fn).to_pandas()

    return df_edges, df_nodes


def load_inputs(basedir: Path, city, split="train", day: Optional[str] = None, df_filter: DF_FILTER = df_filter_weekdays_daytime_only) -> pd.DataFrame:
    """Helper for loading input data (vehicle counts on.

    Parameters
    ----------
    basedir: data basedir
    city: "london" / "madrid" / "melbourne"
    split: "train" / "test" / ...
    day: date as string; if None, loads all files
    df_filter: filter taking data frame as input and returning a filtered data frame.

    Returns
    -------
    """
    infix = "" if day is None else f"_{day}"
    fn = basedir / split / city / "input" / f"counters{infix}.parquet"
    df = pq.read_table(fn).to_pandas()
    if df_filter is not None:
        df = df_filter(df)
    return df


def load_cc_labels(
    basedir: Path, city, split="train", day=None, df_filter: DF_FILTER = df_filter_weekdays_daytime_only, with_edge_attributes: bool = False
) -> pd.DataFrame:
    """Helper for laoding cc labels from basedir.

    Optionally for given day only, optionally filtering, and optionally
    merging with edges.
    """
    if split == "test":
        datadir = basedir / "withheld" / "golden" / city / "labels"
    else:
        datadir = basedir / split / city / "labels"
    if day is not None:
        fn = datadir / f"cc_labels_{day}.parquet"
        df = pq.read_table(fn).to_pandas()
    else:
        dfs = [pq.read_table(fn).to_pandas() for fn in datadir.rglob("cc_labels*.parquet")]
        df = pd.concat(dfs)
    if df_filter is not None:
        df = df_filter(df)
    if with_edge_attributes:
        df_edges, df_nodes = load_road_graph(basedir, city)
        df = df.merge(df_edges, on=["u", "v"], how="left", suffixes=("", "_"))
    return df


def cc_dates(basedir: Path, city, split="train") -> List[str]:
    """Load dates for which there congestion data."""
    return [re.search("[0-9]{4}-[0-9]{2}-[0-9]{2}", str(fn)).group() for fn in (basedir / split / city / "input").rglob("counters_*.parquet")]


# -----------------------------------------------------------------------------------------------------
# DETERMINE CLASS FRACTIONS FOR CONGESTION CLASSES
# -----------------------------------------------------------------------------------------------------


# derived through `run_cc_distribution(split="train")`, see `exploration/cc_distribution.ipynb`
# this is from the participants' training data
class_fractions = {
    "london": ({"green": 0.5367906303432076, "yellow": 0.35138063340805714, "red": 0.11182873624873524}),
    "madrid": {"green": 0.4976221039083026, "yellow": 0.3829591430424158, "red": 0.1194187530492816},
    "melbourne": {"green": 0.7018930324884697, "yellow": 0.2223245729555099, "red": 0.0757823945560204},
}
