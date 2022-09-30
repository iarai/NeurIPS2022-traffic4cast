# Data Specification

## `road_graph`

### `road_graph/<CITY>/road_graph_edges.parquet`

| Key | Attribute     | Example      | Data Type | Description                                                                                                                                                                |
|-----|---------------|--------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u                | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                             |
| x   | v                | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                             |
|     | speed_kph        | 32.2 | double | maxspeed parsed by osmnx (see https://osmnx.readthedocs.io/en/stable/osmnx.html?highlight=speed_kph#osmnx.speed.add_edge_speeds)                                           |
|     | parsed_maxspeed  | 32.2 | double | same as `speed_kph` except for Madrid where there are parsing errors, so here we parse the `maxspeed` field from OSM, see https://wiki.openstreetmap.org/wiki/Key:maxspeed |
|     | importance       | 0 | int64 | numerical  mapping of the osm highway class from 5 (highway), 4 (trunk), 3 (primary), 2 (secondary), 1 (tertiary) and 0 (anything else)                                    |
|     | highway          | unclassified | string | See https://wiki.openstreetmap.org/wiki/Key:highway                                                                                                                        |
|     | oneway           | False | bool | See https://wiki.openstreetmap.org/wiki/Key:oneway                                                                                                                         |
|     | lanes            |  | string | See https://wiki.openstreetmap.org/wiki/Key:lanes                                                                                                                          |
|     | tunnel           |  | string | See https://wiki.openstreetmap.org/wiki/Key:tunnel                                                                                                                         |
|     | length_meters    | 19.402385843568535 | double | edge length in meters                                                                                                                                                      |
|     | counter_distance | 6 | int64 | number of node hops to get to the closes vehicle counter in the graph                                                                                                      |

### `road_graph/<CITY>/road_graph_nodes.parquet`

| Key | Attribute     | Example      | Data Type | Description                                      |
|-----|---------------|--------------|-----------|--------------------------------------------------|
| x   | node_id | 78112 | int64 | OSM ID                                           |
|     | counter_info | ['17/116']                                                                 | list<item: string> | List of loop counter IDs in this node, referring to the city-specific sources.                                                     |
|     | num_assigned | ['1']                                                                      | list<item: string> | number of road graph nodes this counter belongs to. This is only used  for large junctions in Melbourne where a single counter value is associated with multiple road graph nodes      |
|     | x | -0.1457924 | double | longitude                                         |
|     | y | 51.526976 | double | latitude                                        |

### `road_graph/<CITY>/road_graph_supersegments.parquet`

| Key | Attribute  | Example                                           | Data Type         | Description                                                                                                                                                                                         |
|-----|------------|---------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | identifier | 1831647232,253033449                              | string                 | Supersegment identifier. We use the first and last node to have a descriptive name since by construction there is at most supersegment between any two nodes.                                       |
|     | nodes      | [1831647232, 316002729, 330347868,...,253033449] | list<item: int64> | List of <int64>. First is `u` and last is `v`. Each is foreign key to `node_id` in `road_graph_nodes.parquet`. Consecutive items foreign key to foreign key to `u,v` in `road_graph_edges.parquet`. |

### `road_graph/<CITY>/cell_mapping.parquet`

| Key | Attribute     | Example      | Data Type | Description |
|-----|---------------|--------------|-----------|-------------|
| x   | u      | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
| x   | v   | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
|     | cells  | [(172, 223, 2, 0.0), (173, 223, 2, 1.0), (173, 223, 3, 1.0), (172, 223, 3, 0.0)] | string | List of `(row, column, heading, fraction)` where `row,column,heading` denotes one of the (495,436,4) directed cells and `fraction` is represents percentage of the length of the segment overlapping with this cell. This can be zero as we add data from neighboring cells closeby.  |

## `speed_classes`

### `speed_classes/<CITY>/speed_classes_<YYYY-mm-dd>.parquet`

| Key | Attribute     | Example      | Data Type | Description                                                                                                                                                                                  |
|-----|---------------|--------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u                | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                               |
| x   | v                | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                               |
| x   | day              | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), same as in file name                                     |
| x   | t                | 9 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                                                                  |
|     | volume_class     | 3 | int64 | 1 for volumes 1 and 2; 3 for volumes 3 and 4; 5 for 5 and above ... note that this applies to normalized volumes hence 1 means the lowest registered volume according to privacy regulations |
|     | median_speed_kph | 19.764705882352942 | double | the median speed  in the 15 minute interval `(day,t)`                                                                                                                                        |
|     | free_flow_kph    | 36.352941176470594 | double | the free flow speed of this edge derived from the dynamic data (this will be the same for all    `(day,t)`      of the same edge                                                             |

## `movie`

### `movie/<CITY>/<YYYY-mm-dd>_<CITY>_8ch.h5`

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

## train

### `train/<CITY>/input/counters_<YYYY-mm-dd>.parquet`

| Key | Attribute     | Example      | Data Type | Description                                                                                                                                                        |
|-----|---------------|--------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | node_id | 10028711 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                     |
| x   | day | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), same as in file name             |
| x   |  t | 4 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                                        |
|     | volumes_1h | [56. 44. 40. 31.] | list<item: double> | The 4 volume counts from 15 minutes each. As this can result from averaging multiple channels in the raw loop counter data, this can be non-integer. Can be `NaN`. |

### `train/<CITY>/labels/cc_labels_<YYYY-mm-dd>.parquet`

| Key | Attribute     | Example      | Data Type | Description                                                                                                                                            |
|-----|---------------|--------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | u | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                         |
| x   | v | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                         |
| x   | day | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), same as in file name |
| x   | t | 9 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                            |
|     | cc | 2 | int64 | range [0,..,3], 0=undefined, 1=green/uncongested, 2=yellow/slowed-down, 3=red/congested                                                                |

### `train/<CITY>/labels/eta_labels_<YYYY-mm-dd>.parquet`

| Key | Attribute | Example   | Data Type | Description                                                                                                                                              |
|-----|-----------|-----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| x   | identifier | 1_3      | string                 | Supersegment identifier. Foreign key to `road_graph_supersegments.parquet`                                                                               |
| x   | day | 2019-07-01 | string    | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), same as in file name |
| x   | t   | 9         | int64     | 15-minute time slot in the range [0,...,96)                                                                                                              |
|     | eta | 66.0      | double    | non-negative estimated travel time over the super-segment                                                                                                |

## `loop_counter`

### `loop_counter/<CITY>/counters_daily_by_node.parquet`

| Key | Attribute     | Example                                                                    | Data Type | Description                                                                                                                       |
|-----|---------------|----------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------|
| x   | node_id      | 10028711                                                                   | string | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                    |
| x   | day          | 2019-07-01                                                                 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) |
|     | counter_info | ['17/116']                                                                 | list<item: string> | Same as `counter_info` in  `road_graph_nodes.parquet` for the same `node_id`                                                  |
|     | num_assigned | ['1']                                                                      | list<item: string> | Same as `num_assigned` in  `road_graph_nodes.parquet` for the same `node_id`                                                                                                 |
|     | volume       | [ 56. 44. 40. 31. 28. 22. ...] | list<item: double> | 96 float values, covering 15 minutes each, can be `NaN`                                                                           |

## `test`

### `test/<CITY>/input/counters_test.parquet` (core competition: congestion classes (cc) and extended competition (eta))

We use the same situations for core (`cc`) and extended (`eta`) competitions!

| Key | Attribute     | Example      | Data Type | Description                                                    |
|-----|---------------|--------------|-----------|----------------------------------------------------------------|
| x   | node_id    | 101818 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| x   | test_idx   | 0 | int64 | The index for the test in range [0,...,100) for each city.     |
|     | volumes_1h | [290. 284. 313. 311.] | list<item: double> | the counts for the 4 15-minute intervals                       |

## `submission`

### `submission/submission_core.zip/<CITY>/cc_labels_test.parquet`

The submission zip for the core competition must have the following folder structure:

```
london/labels/cc_labels_test.parquet
madrid/labels/cc_labels_test.parquet
melbourne/labels/cc_labels_test.parquet
```

| Key | Attribute | Example | Data Type | Description                                                    |
|-----|---------------|--------------|-----------|----------------------------------------------------------------|
| x   | u            | 99936 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| x   | v            | 2146383887 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| x   | test_idx     | 0 | int64 | The index for the test in range [0,...,100) for each city.     |
|     | logit_green  | -3.551521 | double | the logit for congestion class 1=green                         |
|     | logit_yellow | -2.632999 | double | the logit for congestion class 2=yellow                        |
|     | logit_red    | -3.558682 | double | the logit for congestion class 3=red                           |

## `submission/submission_extended.zip/<CITY>/eta_labels_test.parquet`

The submission zip for the extended competition must have the following folder structure

```
london/labels/eta_labels_test.parquet
madrid/labels/eta_labels_test.parquet
melbourne/labels/eta_labels_test.parquet
```

| Key | Attribute | Example | Data Type | Description                                                                     |
|-----|-----------|-------------|-----------|---------------------------------------------------------------------------------|
| x   | identifier | 1_3      | string                 | Supersegment identifier. Foreign key to `road_graph_supersegments.parquet` |
| x   | test_idx  | 0 | int64     | The index for the test in range [0,...,100) for each city.                      |
|     | eta       | 3.551521 | double    | ETA for the supersegment. Must be non-negative, in particular must not be `NaN`. |

## `withheld`

Just for illustrative purposes on how the evaluation is performed. Participants do not have access to the withheld data.

### `withheld/golden/<CITY>/labels/cc_labels_test.parquet`

| Key | Attribute     | Example      | Data Type | Description                                                                             |
|-----|---------------|--------------|-----------|-----------------------------------------------------------------------------------------|
| x   | u        | 99936 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                          |
| x   | v        | 2146383887 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                          |
| x   | test_idx | 0 | int64 | The index for the test in range [0,...,100) for each city.                              |
|     | cc       | 2 | int64 | range [0,..,3], 0=undefined, 1=green/uncongested, 2=yellow/slowed-down, 3=red/congested |

### `withheld/golden/<CITY>/labels/eta_labels_test.parquet`

| Key | Attribute | Example      | Data Type | Description                                                    |
|-----|-----------|--------------|-----------|----------------------------------------------------------------|
| x   | identifier | 1_3      | string                 | Supersegment identifier. Foreign key to `road_graph_supersegments.parquet` |
| x   | test_idx | 0 | int64 | The index for the test in range [0,...,100) for each city.     |
|     | eta      | -3.551521 | double | ETA for the supersegment. Can be `NaN`.                         |
