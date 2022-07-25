# Traffic4cast 2022 Exploration

## `data_exploration`

Explore the data, how to load and visualize.

## `example_static_plain_torch_dataset`

Loading from plain torch dataset, applying static strategies (uniform or always green/yellow/red) and evaluating.

## `example_torch_geometric_dummy_GNN`

Loading from torch geometric dataset
A dense node embedding of the spatially sparse volumes, followed by a stack of `MessagePassing` layers, and finally, a stack of dense layers predicting green,
yellow, red distribution from the two nodes of each edge separately, trained end to end.

## `data_inventory`

Inventory of the `.parquet` files and their schemas.
