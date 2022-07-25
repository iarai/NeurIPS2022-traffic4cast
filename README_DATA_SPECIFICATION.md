# Data Specification

## `train/london/input/counters_2019-07-01.parquet`

| Attribute     | Example      | Data Type | Description                                                                                                                                                        |
|---------------|--------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| node_id | 10028711 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                     |
| day | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)                                 |
| t | 4 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                                        |
| volumes_1h | [56. 44. 40. 31.] | list<item: double> | The 4 volume counts from 15 minutes each. As this can result from averaging multiple channels in the raw loop counter data, this can be non-integer. Can be `NaN`. |

## `train/london/labels/cc_labels_2019-07-01.parquet`

| Attribute     | Example      | Data Type | Description                                                                                                                       |
|---------------|--------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------|
| u | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                    |
| v | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                    |
| day | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) |
| t | 9 | int64 | 15-minute time slot in the range [0,...,96)                                                                                       |
| cc | 2 | int64 | range [1,..,4], 1=green/uncongested, 2=yellow/slowed-down, 3=red/congested                                        |

## `road_graph/london/cell_mapping.parquet`

| Attribute     | Example      | Data Type | Description |
|---------------|--------------|-----------|-------------|
| u | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
| v | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`   |
| cells | [(172, 223, 2, 0.0), (173, 223, 2, 1.0), (173, 223, 3, 1.0), (172, 223, 3, 0.0)] | string | List of `(row, column, heading, fraction)` where `row,column,heading` denotes one of the (495,436,4) directed cells and `fraction` is represents percentage of the length of the segment overlapping with this cell. This can be zero as we add data from neighboring cells closeby.  |

## `road_graph/london/road_graph_edges.parquet`

| Attribute     | Example      | Data Type | Description                                                                                                                                                                |
|---------------|--------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| u | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                             |
| v | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                             |
| speed_kph | 32.2 | double | maxspeed parsed by osmnx (see https://osmnx.readthedocs.io/en/stable/osmnx.html?highlight=speed_kph#osmnx.speed.add_edge_speeds)                                           |
| parsed_maxspeed | 32.2 | double | same as `speed_kph` except for Madrid where there are parsing errors, so here we parse the `maxspeed` field from OSM, see https://wiki.openstreetmap.org/wiki/Key:maxspeed |
| importance | 0 | int64 | numerical  mapping of the osm highway class from 5 (highway), 4 (trunk), 3 (primary), 2 (secondary), 1 (tertiary) and 0 (anything else)                                    |
| highway | unclassified | string | See https://wiki.openstreetmap.org/wiki/Key:highway                                                                                                                        |
| oneway | False | bool | See https://wiki.openstreetmap.org/wiki/Key:oneway                                                                                                                         |
| lanes |  | string | See https://wiki.openstreetmap.org/wiki/Key:lanes                                                                                                                          |
| tunnel |  | string | See https://wiki.openstreetmap.org/wiki/Key:tunnel                                                                                                                         |
| length_meters | 19.402385843568535 | double | edge length in meters                                                                                                                                                      |
| counter_distance | 6 | int64 | number of node hops to get to the closes vehicle counter in the graph                                                                                                      |

## `road_graph/london/road_graph_nodes.parquet`

| Attribute     | Example      | Data Type | Description                                      |
|---------------|--------------|-----------|--------------------------------------------------|
| node_id | 78112 | int64 | OSM ID                                           |
| counter_info | ['17/116']                                                                 | list<item: string> | List of loop counter IDs in this node, referring to the city-specific sources.                                                     |
| num_assigned | ['1']                                                                      | list<item: string> | number of road graph nodes this counter belongs to. This is only used  for large junctions in Melbourne where a single counter value is associated with multiple road graph nodes      |
| x | -0.1457924 | double | latitude                                         |
| y | 51.526976 | double | longitude                                        |

## `loop_counter/london/counters_daily_by_node.parquet`

| Attribute     | Example                                                                    | Data Type | Description                                                                                                                       |
|---------------|----------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------|
| node_id | 10028711                                                                   | string | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                    |
| day | 2019-07-01                                                                 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) |
| counter_info | ['17/116']                                                                 | list<item: string> | Same as `counter_info` in  `road_graph_nodes.parquet` for the same `node_id`                                                  |
| num_assigned | ['1']                                                                      | list<item: string> | Same as `num_assigned` in  `road_graph_nodes.parquet` for the same `node_id`                                                                                                 |
| volume | [ 56. 44. 40. 31. 28. 22. ...] | list<item: double> | 96 float values, covering 15 minutes each, can be `NaN`                                                                           |





## `speed_classes/london/speed_classes_2019-07-01.parquet`

| Attribute     | Example      | Data Type | Description                                                                                                                                                                                  |
|---------------|--------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| u | 78112 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                               |
| v | 25508583 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                                                                                                                               |
| day | 2019-07-01 | string | formatted string `%Y-%m-%d` see [format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)                                                           |
| t | 9 | int64 | 15-minute time slot in the range [0,...,96)                                                                                                                                                  |
| volume_class | 3 | int64 | 1 for volumes 1 and 2; 3 for volumes 3 and 4; 5 for 5 and above ... note that this applies to normalized volumes hence 1 means the lowest registered volume according to privacy regulations |
| median_speed_kph | 19.764705882352942 | double | the median speed  in the 15 minute interval `(day,t)`                                                                                                                                        |
| free_flow_kph | 36.352941176470594 | double | the free flow speed of this edge derived from the dynamic data (this will be the same for all    `(day,t)`      of the same edge                                                             |

## `movie/london/2019-07-01_london_8ch.h5`

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
| 6 | volume NE |
| 7 | speed NE |

## `test/london/input/counters_test.parquet`

**SUBJECT TO CHANGE, WILL BE FINALIZED WHEN LEADERBOARD OPENS**

| Attribute     | Example      | Data Type | Description                                                    |
|---------------|--------------|-----------|----------------------------------------------------------------|
| node_id | 101818 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| volumes_1h | [290. 284. 313. 311.] | list<item: double> | the counts for the 4 15-minute intervals                       |
| test_idx | 0 | int64 | The index for the test in range [0,...,200) for each city.     |

## `submission/london/cc_labels_test.parquet`

**SUBJECT TO CHANGE, WILL BE FINALIZED WHEN LEADERBOARD OPENS**

| Attribute | Example | Data Type | Description                                                    |
|---------------|--------------|-----------|----------------------------------------------------------------|
| u | 99936 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| v | 2146383887 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet` |
| logit_green | -3.551521 | double | the logit for congestion class 1=green                         |
| logit_yellow | -2.632999 | double | the logit for congestion class 2=yellow                        |
| logit_red | -3.558682 | double | the logit for congestion class 3=red                           |
| test_idx | 0 | int64 | The index for the test in range [0,...,200) for each city.     |
|

## `withheld/golden/london/labels/cc_labels_test.parquet`

**SUBJECT TO CHANGE, WILL BE FINALIZED WHEN LEADERBOARD OPENS**

| Attribute     | Example      | Data Type | Description                                                                              |
|---------------|--------------|-----------|------------------------------------------------------------------------------------------|
| u | 99936 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                           |
| v | 2146383887 | int64 | OSM ID, foreign key to `node_id` in `road_graph_nodes.parquet`                           |
| cc | 2 | int64 | range [1,..,4], 1=green/uncongested, 2=yellow/slowed-down, 3=red/congested |
| test_idx | 0 | int64 | The index for the test in range [0,...,200) for each city.     |
