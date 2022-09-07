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
import argparse
import datetime
import logging
import os
import re
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from t4c22.dataloading.t4c22_dataset import T4c22Competitions

# London 	2019-07-01 - 2020-01-31
# Madrid 	2021-06-01 - 2021-12-31
# Melbourne 	2020-06-01 - 2020-12-30
train_start_dates = {  # noqa
    "london": "2019-07-06",  # 5
    "madrid": "2021-06-01",  # 1
    "melbourne": "2020-06-01",  # 0
}
test_start_dates = {  # noqa
    "london": "2019-12-01",
    "madrid": "2021-12-01",
    "melbourne": "2020-12-01",
}

groupings_cc = {
    "hour": (["u", "v", "h"], lambda df: df),
    "allday": (["u", "v"], lambda df: df),
    "daytime": (["u", "v"], lambda df: df[(df["t"] >= 6 * 4) & (df["t"] < 22 * 4)]),
}

groupings_eta = {
    "hour": (["identifier", "h"], lambda df: df),
    "allday": (["identifier"], lambda df: df),
    "daytime": (["identifier"], lambda df: df[(df["t"] >= 6 * 4) & (df["t"] < 22 * 4)]),
}

weekday_filters = {
    # "Monday": lambda date: date.weekday()==0, # noqa
    # "Tuesday": lambda date: date.weekday()==1, # noqa
    # "Wednesday": lambda date: date.weekday()==2, # noqa
    # "Thursday": lambda date: date.weekday()==3, # noqa
    # "Friday": lambda date: date.weekday()==4, # noqa
    # "Saturday": lambda date: date.weekday()==5, # noqa
    # "Sunday": lambda date: date.weekday()==6, # noqa
    "workdays": lambda df: df[df["weekday"] < 5],
    "weekend": lambda df: df[df["weekday"] >= 5],
    "all_week": lambda df: df,
}
DateFilter = Callable[[datetime.date], bool]


def load_train_labels(data_folder: Path, city: str, competition: str):
    train_label_frames = []
    train_label_files = sorted((data_folder / "train" / city / "labels").glob(f"{competition}_labels_*.parquet"))
    for train_label_file in tqdm.tqdm(train_label_files, total=len(sorted(train_label_files))):
        df = pd.read_parquet(train_label_file)
        day_str = re.search("[0-9]{4}-[0-9]{2}-[0-9]{2}", str(train_label_file)).group()
        day = datetime.datetime.strptime(day_str, "%Y-%m-%d")
        df["day"] = day_str
        df["weekday"] = day.weekday()
        df["h"] = df["t"] // 4
        df["timestamp"] = [day + datetime.timedelta(hours=t // 4, minutes=(t % 4) * 15) for t in df["t"]]
        train_label_frames.append(df)
    logging.info(f"Read {len(train_label_frames)} training label files for {city} {competition}")
    train_labels = pd.concat(train_label_frames)
    logging.info(f"Labels loaded: {len(train_labels)} for {city} {competition}")
    return train_labels


def get_data(data_folder: Path, city: str, competition="cc"):
    df = load_train_labels(data_folder=data_folder, city=city, competition=competition)
    if competition == "cc":
        df_edges = pd.read_parquet(data_folder / "road_graph" / city / "road_graph_edges.parquet")
        merge_keys = ["u", "v"]
    elif competition == "eta":
        merge_keys = ["identifier"]
        df_edges = pd.read_parquet(data_folder / "road_graph" / city / "road_graph_supersegments.parquet")
    else:
        raise
    df = df.merge(df_edges, left_on=merge_keys, right_on=merge_keys, how="left")
    return df


def historic_distribution_for_city(args, data_folder: Path):
    competition, city = args
    logging.info(f"/ start {city} {competition} {psutil.virtual_memory()}")
    df = get_data(data_folder=data_folder, city=city, competition=competition)
    for weekday_label, weekday_filter in weekday_filters.items():
        logging.info(f"/ start {city} {competition} {weekday_label} {psutil.virtual_memory()}")
        df_filtered = weekday_filter(df)
        logging.info(f"{len(df)} -> {len(df_filtered)} for {weekday_label}")
        if competition == "cc":
            assert len(df[df["cc"].isna()]) == 0, (data_folder, df[df["cc"].isna()])
            _historic_cc(data_folder, city, df_filtered, weekday_label)
        elif competition == "eta":
            _historic_eta(data_folder, city, df_filtered, weekday_label)
        else:
            raise
        logging.info(f"\\ end {city} {competition} {weekday_label} {psutil.virtual_memory()}")
    logging.info(f"\\ end {city} {competition} {psutil.virtual_memory()}")


def _historic_cc(BASEDIR, city, df, weekday_label):
    logging.info(f"/ start {city} {weekday_label} {psutil.virtual_memory()}")
    assert "weekday" in df.columns
    assert df["cc"].min() >= 0, df["cc"].min()
    assert df["cc"].max() == 3, df["cc"].max()
    assert len(df[df["cc"] == 0]) + len(df[df["cc"] == 1]) + len(df[df["cc"] == 2]) + len(df[df["cc"] == 3]) == len(df)
    # ## Compute distribution of CC for each time of day and w / w/o day of week
    for grouping, (grouping_keys, grouping_filter) in tqdm.tqdm(groupings_cc.items()):
        logging.info(f"/ start {city} cc {weekday_label} {grouping} - {grouping_keys}")
        df_filtered = grouping_filter(df)
        df_grouped = df_filtered.groupby(grouping_keys + ["cc"]).count().reset_index()
        # take parsed_maxspeed, any other column not used for grouping could have been used.
        assert df_grouped["parsed_maxspeed"].sum() == len(df_filtered), (df_grouped["h"].sum(), len(df_filtered))
        df_counts = df_grouped.pivot(index=grouping_keys, columns="cc", values="parsed_maxspeed")
        df_counts = df_counts.fillna(0)
        assert df_counts[1].sum() == len(df_filtered[df_filtered["cc"] == 1])
        assert df_counts[2].sum() == len(df_filtered[df_filtered["cc"] == 2])
        assert df_counts[3].sum() == len(df_filtered[df_filtered["cc"] == 3])
        df_counts["total"] = df_counts[1] + df_counts[2] + df_counts[3]
        df_counts["proba_green"] = df_counts[1] / df_counts["total"]
        df_counts["proba_yellow"] = df_counts[2] / df_counts["total"]
        df_counts["proba_red"] = df_counts[3] / df_counts["total"]
        p = Path(f"{BASEDIR}/congestion_classes_distribution/{city.lower()}/")
        p.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df_counts)

        fn = p / f"congestion_classes_distribution_{grouping}_{weekday_label}.parquet"
        pq.write_table(table, fn, compression="snappy")
        logging.info(f"written {fn}")
        assert len(pq.read_table(fn).to_pandas()) == len(df_counts), (len(pq.read_table(fn).to_pandas()), len(df_counts))
        logging.info(f"\\ end {city} cc {weekday_label} {grouping} - {grouping_keys}")
        del table
        # del df_counts
        # del df_filtered
    logging.info(f"\\ end {city} {weekday_label} {psutil.virtual_memory()}")


def _historic_eta(BASEDIR, city, df, weekday_label):
    assert "weekday" in df.columns
    assert df["eta"].min() >= 0
    assert df["eta"].isna().sum() == 0, (city, weekday_label, df["eta"].isna().sum())

    # ## Compute distribution of ETA for each time of day and w / w/o day of week
    for grouping, (grouping_keys, grouping_filter) in tqdm.tqdm(groupings_eta.items()):
        logging.info(f"/ start {city} eta {weekday_label} {grouping} - {grouping_keys}")
        df_filtered = grouping_filter(df)
        df_grouped = df_filtered.groupby(grouping_keys)
        df_counts = df_grouped.agg(
            eta_min=("eta", np.min),
            eta_max=("eta", np.max),
            eta_mean=("eta", np.mean),
            eta_med=("eta", np.median),
            # eta_q10=('eta', partial(np.quantile,q=0.1)),  # noqa
            # eta_q90=('eta', partial(np.quantile,q=0.9)), # noqa
            eta_std=("eta", np.std),
        )

        p = Path(f"{BASEDIR}/expected_time_of_arrival_distribution/{city.lower()}/")
        p.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df_counts)

        fn = p / f"expected_time_of_arrival_distribution_{grouping}_{weekday_label}.parquet"
        pq.write_table(table, fn, compression="snappy")
        logging.info(f"written {fn}")
        assert len(pq.read_table(fn).to_pandas()) == len(df_counts), (len(pq.read_table(fn).to_pandas()), len(df_counts))
        logging.info(f"\\ end {city} {weekday_label} {grouping} - {grouping_keys}")
        del table
        del df_counts
        del df_filtered
    logging.info(f"\\ end {city} eta {weekday_label} {psutil.virtual_memory()}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("This script takes the T4c22 data directory and checks the training label files."))
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data",
        required=True,
    )
    parser.add_argument("-c", "--competition", type=T4c22Competitions, help="Competition", required=False, default=["cc", "eta"], nargs="+")
    parser.add_argument("--city", type=str, help="Competition", required=False, default=["london", "madrid", "melbourne"], nargs="+")
    parser.add_argument("-j", "--jobs", type=int, help="Number of jobs to run in parallel", required=False, default=1)
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
        jobs = params["jobs"]
        cities = params["city"]
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        raise e
    if jobs > 1:
        with Pool(processes=jobs) as pool:
            _ = list(
                pool.imap_unordered(
                    partial(historic_distribution_for_city, data_folder=data_folder), [(competition, city) for competition in competitions for city in cities]
                )
            )
    else:
        for competition in competitions:
            for city in cities:
                historic_distribution_for_city(args=(competition, city), data_folder=data_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
