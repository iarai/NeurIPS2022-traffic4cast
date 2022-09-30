# Data Specification Data Pipeline

Describes the data format for each of the data pipeline steps.

## HERE Traffic Map Movies

### `movie/<CITY>/2019-04-30_<CITY>_8ch.h5`

SAME as t4c22, see https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md

* dtype: `uint8`
* shape: `(288,495,436,8)`
    * 5 minute bin, 288 per day
    * 495 rows
    * 436 columns
    * 8 channels:

| Channel     | Description |
|-------------|-------------|
| 0 | volume NE |
| 1 | speed NE |
| 2 | volume NW |
| 3 | speed NW |
| 4 | volume SE |
| 5 | speed SE |
| 6 | volume SW |
| 7 | speed SW |

## `dp01_movie_aggregation`

### `movie_15min/<CITY>/<YYYY-mm-dd>_<CITY>_8ch_15min.h5`

* dtype: `float64`
* shape: `(96, 495, 436, 8)`
    * 15 minute bin, 96 per day
    * 495 rows
    * 436 columns
    * 8 channels see above

## `dp02_speed_clusters`

### `movie_speed_clusters/<CITY>/speed_clusters.h5`

* dtype: `float64`
* shape: `(495, 436, 4, 5, 2)`
    * 495 rows
    * 436 columns
    * 4 headings `NE`, `NW`, `SE`, `SW`
    * 5 speed clusters
    * 2: cluster median and cluster size

## `dp03_road_graph`

### `road_graph/<CITY>/road_graph.graphml`

As downloaded through `osmnx` [`graph_from_bbox`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox) with
default  `network_type="drive"`, `truncate_by_edge=True` and by default `custom_filter=["highway"~"motorway|motorway_link|trunk|primary|secondary|tertiary"]`
for the city bounding box,
with edge speeds and edge travel times added through `osmnx` [`add_edge_speeds`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.speed.add_edge_speeds)
and [`add_edge_travel_times`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.speed.add_edge_travel_times), respectively. A custom filter can be passed
through `--custom_filter filter_string` to `dp03_road_graph.py`.

### `road_graph/<CITY>/road_graph.gpkg`

The graphml saved as geopackage through `osmnx` [`save_graph_geopackage`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.io.save_graph_geopackage)
containing two layers `nodes` and `edges` for easy visualization in tools like [QGIS](https://www.qgis.org/).

### `road_graph/<CITY>/road_graph_edges.parquet`

SAME as t4c22, see https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md with fewer columns (`parsed_maxspeed`, `importance` , `counter_distance`).


| Key | Attribute     | Example      | Data Type | Description                                                                                                                                                                                                                                                                                                                     |
|-----|---------------|--------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u                | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                                                                                                                                                  |
| x   | v                | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                                                                                                                                                  |
|     | speed_kph        | 32.2 | double | parsed maxspeed value, by default this uses osmnx to parse (see https://osmnx.readthedocs.io/en/stable/osmnx.html?highlight=speed_kph#osmnx.speed.add_edge_speeds), with the flag `--parse_maxspeed` a custom logic is used which avoids using averages in the case of multiple speed limits (this is recommended e.g. in Madrid) |
|     | maxspeed         | 32.2 | string | Unparsed maxspeed string value as stored in the OSM flag, see https://wiki.openstreetmap.org/wiki/Key:maxspeed                                                                                                                                                                                                                  |
|     | highway          | unclassified | string | See https://wiki.openstreetmap.org/wiki/Key:highway                                                                                                                                                                                                                                                                             |
|     | oneway           | False | bool | See https://wiki.openstreetmap.org/wiki/Key:oneway                                                                                                                                                                                                                                                                              |
|     | lanes            |  | string | See https://wiki.openstreetmap.org/wiki/Key:lanes                                                                                                                                                                                                                                                                               |
|     | tunnel           |  | string | See https://wiki.openstreetmap.org/wiki/Key:tunnel                                                                                                                                                                                                                                                                              |
|     | length_meters    | 19.402385843568535 | double | edge length in meters                                                                                                                                                                                                                                                                                                           |
|     | osmid | [622326987, 673699601, 147432151, 147432152, 1071291898, 4610047] | string | OSM ID or list of OSM IDs as string. This can be several nodes as the graph is simplified by default in [`graph_from_bbox`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox). |
|     | geometry | b'\x01\x02\x00\x00\x00?\x00\x00\x00\x94\x84....'                  | binary | Geometries as resulting from `osmnx` [`graph_from_gdfs`](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_graph.graph_from_gdfs)                                                               |

### `road_graph/<CITY>/road_graph_nodes.parquet`

SAME as t4c22, see https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md, except for fewer columns (`counter_info`
and `num_assigned`)

| Attribute     | Example      | Data Type | Description |
|---------------|--------------|-----------|-------------|
| node_id | 172539.0 | int64 |    |
| x | 13.3355015 | double |    |
| y | 52.5652055 | double |    |

## `dp04_intersecting_cells`

### `road_graph/<CITY>/road_graph_intersecting_cells.parquet`

SAME as t4c22, see https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md

| Key | Attribute     | Example      | Data Type | Description |
|-----|---------------|--------------|-----------|-------------|
| x   | u      | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
| x   | v   | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
|     | speed_kph<br/> highway <br/>oneway <br/>lanes <br/>tunnel <br/>length_meters |  |  |  inherited fields from `road_graph_intersecting_cells.parquet`     |
|     | intersecting_cells  | [(172, 223, 2, 0.0), (173, 223, 2, 1.0), (173, 223, 3, 1.0), (172, 223, 3, 0.0)] | string | List of `(row, column, heading, fraction)` where `row,column,heading` denotes one of the (495,436,4) directed cells and `fraction` is represents percentage of the length of the segment overlapping with this cell. This can be zero as we add data from neighboring cells closeby.  |

## `dp05_free_flow`

### `road_graph/<CITY>/road_graph_freeflow.parquet`

| Key | Attribute     | Example                                                                      | Data Type | Description                                                                                                                                                                                                                                                                          |
|-----|---------------|------------------------------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u | 172539                                                                       | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                                                                                                       |
| x   | v | 28255136                                                                     | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                                                                                                       |
| x   | gkey             | -6658749479554183131	 | int64 | Hash key of the segment geometry. Unique identifier of the segment also in case of circular roads with same start and end.                                                                        |
|     | inherited fields from road_graph_intersecting_cells.parquet |  |  |  |
|     | free_flow_kph | 48.94117647058823                                                            | double | the free flow speed of this edge derived from the dynamic data. Capped by `speed_kph` if `--use_speed_limit` is set in `. Will be `-1.0` if                                                                                                                                          |

## `dp06_speed_classes`

### `speed_classes/<CITY>/speed_classes_<YYYY-mm-dd>.parquet`

SAME as t4c22, see https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md

If there is not enough data to derive ̀median_speed_kph` for an edge `(u,v)` at `(day,t)`, there will be no row for it.

| Key | Attribute     | Example      | Data Type | Description                                                                                                                                                                                            |
|-----|---------------|--------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u                | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                         |
| x   | v                | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                                         |
| x   | gkey             | -6658749479554183131	 | int64 | Hash key of the segment geometry. Allows joining with geometry and other edge metadata from road_graph_freeflow.parquet                                                                        |
| x   | day              | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), same as in file name                                               |
| x   | t                | 9 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                                                                            |
|     | volume_class     | 3 | int64 | 1 for volumes 1 and 2; 3 for volumes 3 and 4; 5 for 5 and above ... note that this applies to normalized volumes hence 1 means the lowest registered volume according to privacy regulations           |
|     | median_speed_kph | 19.764705882352942 | double | the median speed  in the 15 minute interval `(day,t)`                                                                                                                                                  |
|     | free_flow_kph    | 36.352941176470594 | double | the free flow speed of this edge derived from the dynamic data. Capped by `speed_kph` if `--use_speed_limit` is set in `dp05_free_flow.py`.  This will be the same for all `(day,t)` of the same edge. |
