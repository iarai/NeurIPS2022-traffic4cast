#  Traffic4cast 2022 Competition: from few public vehicle counters to entire city-wide traffic

[![Tests](https://github.com/iarai/NeurIPS2022-traffic4cast/actions/workflows/tests.yaml/badge.svg)](https://github.com/iarai/NeurIPS2022-traffic4cast/actions/workflows/tests.yaml)
[![Library Install](https://github.com/iarai/NeurIPS2022-traffic4cast/actions/workflows/library-install.yaml/badge.svg)](https://github.com/iarai/NeurIPS2022-traffic4cast/actions/workflows/library-install.yaml)
[![Code style: flake8](https://img.shields.io/badge/Code%20style-flake8-yellow.svg)](https://github.com/pycqa/flake8/)
[![Code formatter: black](https://img.shields.io/badge/Code%20formatter-black-000000.svg)](https://github.com/psf/black)

![traffic4cast2020](t4c20logo.png)

* Predict congestion classes (red/yellow/green) and average speed.
* Work with real world data from industrial-sized fleets and vehicle counters.
* Apply Graph Learning to predict traffic dynamics on the road graph edges given sparse node data.

![cc_animation.gif](exploration/cc_animation.gif)

## About
[This repo](https://github.com/iarai/NeurIPS2022-traffic4cast) contains all information about the Traffic4cast 2022 NeurIPS competition for participants. It contains detailed information about the competition and data as well as code and [issue tracker](https://github.com/iarai/NeurIPS2022-traffic4cast/issues).

## Competition Timeline
* 27nd July 2022: training data and code for **Core Challenge** available (see https://www.iarai.ac.at/traffic4cast/challenge/)
* 19th August 2022: test set release and leaderboard opens for **Core Challenge**
* 2nd September 2022: code available, test set release and leaderboard opens for **Extended Challenge**
* **24th** ~15th~ of **October 2022 11:59am CEST**: **Submission** to Traffic4cast 2022 leaderboard deadline
* **31st** ~21st~ of **October 2022 11:59am CET**: **Abstract** submission deadline
* **5th November 2022**: announcement of the **winners**

## TL;DR

### Download Links
- [LONDON_2022.zip](https://developer.here.com/sample-data) from HERE (2.8 GB)
- [MADRID_2022.zip](https://developer.here.com/sample-data) from HERE (4.0 GB)
- [MELBOURNE_2022.zip](https://developer.here.com/sample-data) from HERE (0.9 GB)
- [T4C_INPUTS_2022.zip](https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip) (1.0 GB)
- [T4C_INPUTS_ETA_2022.zip](https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip) (available September 2, 2022, 1.5MB)
- Tests for the core competition are contained in `T4C_INPUTS_2022.zip` above.
- Tests for the extended competition are contained in `T4C_INPUTS_2022.zip` above (same as for core competition).


For more details, see [Folder Structure](#folder-structure) below.

### Prepare environment


```bash
conda env update -f environment.yaml
conda activate t4c22

# Installing the torch geometric extras is optional, required only if using `torch_geometric`
# install-extras is not passed to pip from environment yaml, therefore add as post-step (https://github.com/conda/conda/issues/6805)
# replace with your CUDA version (cpu, ...), see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA="cu113"
python -m pip install -r install-extras-torch-geometric.txt -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html

python t4c22/misc/check_torch_geometric_setup.py
```


### Generate labels
After downloading the data (see [Download Links](#download-links)), run
```bash
prepare_training_data_cc.py -d <data folder with unzipped downloads>
prepare_training_data_eta.py -d <data folder with unzipped downloads>
prepare_training_check_labels.py -d <data folder with unzipped downloads>
```


### Start jupyter notebook and explore

```bash
cd <root of the repo clone>
export PYTHONPATH=<root of the repo clone>

jupyter notebook
Jupyter Notebook 6.4.12 is running at:
http://localhost:8888/?token=xxx
 or http://127.0.0.1:8888/?token=xxx
```

Then open the above link in the browser and have a look at the data exploration and baseline notebooks


## Competition Description
Our Traffic4cast competition series at NeurIPS has contributed both methodological and practical insights to advance the application of AI to forecasting traffic and other spatial processes. Now, going beyond the Traffic4cast challenges at NeurIPS 2019, 2020, and 2021, this year we will explore models that have the ability to **generalize loosely related temporal vertex data on just a few nodes** to **predict dynamic future traffic states on the edges of the entire road graph**. Specifically, in our core challenge we invite participants to predict for three cities the **congestion classes** known from the red, yellow, or green colouring of roads on a common traffic map for the entire road graph 15min into the future. We provide **car count data from spatially sparse vehicle counters** in these three cities in 15min aggregated time bins for one hour prior to the prediction time slot. For our extended challenge participants are asked to predict the **actual average speeds** on each road segment in the graph 15min into the future.


![traffic4cast2022](Main-image-web-08.png)
<img src="./img/task_overview.svg">

Interested in understanding traffic around the globe in a novel way? [Join us](https://www.iarai.ac.at/traffic4cast/forums/) to help shape this year’s competition.


### Task
Given one hour of sparse loop count data only the task is to predict the congestion
classification (`cc`) for all road segments 15 minutes into the future (green, yellow, and red lines). For the
extended challenge the prediction targets are Travel Times or Expected Times of Arrival (`eta`) along supersegments.

Not only the input data is sparse, but also the ground truth labels are sparse. As GPS fleets comprise only a subset of all vehicles in traffic, many segments cannot be classified from the data without further data assimilation (as in production systems).
Models need to be able to cope with such sparse signal during training while still being able to predict accurately whenever ground truth labels are available. See [Metric](#metric) below.

The ability to predict temporal edge features in graphs from sparse vertex data is a key
concern in real world applications more generally, many of which are critical to the functioning of
our society. Our competition affects research solution approaches to many applications, examples are:
the spread of COVID-19 can be seen as predicting temporal edge features in a large spatio-temporal graph;
malicious software can be identified as anomalous patterns in temporal sequences of evolving multiscaled API graph;
Wide Area Network traffic prediction;
learning the temporal dynamics of crypto-currency networks.



### Data
The GPS data has been aggregated and made available by HERE Technologies as in the previous
instances of Traffic4cast. The data originates from a large fleet of probe vehicles which recorded their movements in the years 2019 to
2021. For this year’s competition we focus on the three cities: London, Madrid and Melbourne.
The underlying raw data is similar to the input for commercial traffic maps and routing
products. Similarly to the previous years, the data will be made available for download from HERE
Technologies. Unlike previous years, this data will serve as input for deriving the ground truth labels
for our core and extended challenge. In addition, participants can use the speed data on the graph
edges for jointly modelling traffic or as labels for intermediate tasks. As mentioned below in the
evaluation data description, we are holding back some time intervals for all three cities.
Traffic loop counter data is the new additional data source for Traffic4cast 2022. Loop counters are
often made publicly available by city or motorway authorities and, lately, an increasing collection
of crowd-sourced data through alternative sensor networks is
becoming widely available. Loop counters are spatially sparse (up to 7000 per city)
but do capture all traffic at their locations. They have no volume bias in contrast to the GPS probe
data which, in turn, are collected by vehicle fleets. We focus on the large
publicly available, loop counter data sets for London, Madrid, and Melbourne. The loop counter
measurements are represented as nodes with measured volumes per 15 minute time bin. The nodes
can be attached to the road graph. We are providing pre-compiled derived data sets as well as tools to
convert or easily load the data.


### Labels
The aggregated GPS probe data are used to generate our ground truth labels. Therefore the 5 minute
time bins in our usual spato-temporal data format are getting aggregated to 15 minutes time bins taking the average speed to our loop counter data. We then derive
the following ground truth labels.

#### Core Challenge: Congestion Class (CC)
Labels in the core competition are congestion classes for each segment in the road graph: red/congested, yellow/warning, green/uncongested.
The class is derived through from the aggregated GPS probe data; if not enough data is available to derive the congestion class, it will output
missing value. For ease of use we will provide the training data as a graph representation.
At test time we provide only the loop counter input and evaluate only against locations where data is available to derive the congestion class.
See [Generate Labels](#generate-labels) for the code used to derive congestion classes.

#### Extended Challenge: Travel Time (ETA) for each supersegment
Labels in the extended competition are travel times (or Expected Time of Arrival, ETA) for each supersegment.
The dynamic speed data (GPS probes) is used to derive travel times on the edges of the graph and then summed up to supersegment ETAs.
This is compatible with how industry-standard routing engines are deriving their segment speeds. The use of supersegments is motivated by contraction hierarchies, which are a common strategy in most commercial routing engines.
Also, the use of supersegments helps to make the ETAs derived from the underlying speed data more robust to data outliers.
At test time we provide only the loop counter input.
See [Generate Labels](#generate-labels) for the code used to derive ETAs.


### Novelty
Traffic4cast 2022 seamlessly connects to and builds on our previous competitions at NeurIPS 2019,
2020 and 2021. It does not reuse any data provided in previous competitions and adds data for two
additional hitherto unused cities (London and Madrid).
Key innotiations in this year’s competition are the following.

**Usage of alternative data sources** It is important to note that our competition challenges do not
ask participants to make predictions into the future on the same data source. Both its challenges
invite participants to make traffic state predictions from sparse loop counter data on road graph edges,
i.e. predict the future dynamics of a different data source. Our competition’s aim is to lower the
technological and financial barrier of entry to making future traffic state predictions of entire cities.
Loop counters are an easily available public data source that encodes much of the complex traffic
dynamics we seek.

**Unique scale.** Our competition data is based on $`10^{12}`$ data points and for the first time allows the
scientific community to study the dynamic link between loop counter data time series and the traffic
state dynamics of entire cities rather than small, non-isolated traffic subsystems.
Our competition thus also represents a unique real-world, large scale graph based prediction problem
that should be of interest to the machine learning community.


### Leaderboards

Test data and leaderboard for the Core Challenge will go live on August 19, 2022. The Extended Challenge opens September 2, 2022.
Leaderboards and data are available via the competition website https://www.iarai.ac.at/traffic4cast/.


### Prizes
For both core and extended competitions each:
* Voucher or cash prize worth 5,000 EUR to the participant/team and one free NeurIPS 2022 conference registration (virtual)
* Voucher or cash prize worth 3,000 EUR to the participant/team and one free NeurIPS 2022 conference registration (virtual)
* Voucher or cash prize worth 2,000 EUR to the participant/team and one free NeurIPS 2022 conference registration (virtual)
* Special prizes for creative and innovative solutions tba





## Submission
The leaderboards will be open according to the [Timeline](#timeline) above.
You will need to upload a zip file with your solution through the [Traffic4cast web site](https://www.iarai.ac.at/traffic4cast/) as in previous years.
See [`t4c22/evaluation/README.md`](https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/t4c22/evaluation/README.md) for  code and examples to create submissions.
The details for the format can be found in [`README_DATA_SPECIFICATION.md`](https://github.com/iarai/NeurIPS2022-traffic4cast/blob/main/README_DATA_SPECIFICATION.md).


## Repo sitemap

```
.
├── exploration                    <-- getting started with t4c22
├── data_pipeline                  <-- data and competition preparation pipeline (forthcoming)
├── baselines                      <-- competition baselines (forthcoming)
├── blogposts                      <-- competition blogposts with additional information and discussion
└── t4c22                          <-- code for t4c22 (can be installed as library)
    ├── evaluation                 <-- helpers for creating submissions, code for generating test set and evaluating test set (forthcoming)
    ├── dataloading                <-- data loader for pytorch (plain and geometric)
    ├── misc                       <-- miscellaneous (logging, checks, etc.)
    └── plotting                   <-- plotting helpers
```

## Data
See [Download Links](#download-links) above.

### Overview
The following figure shows an overview of the data for the Traffic4cast 2022 Competition at NeurIPS and how it is used in the competition. For legal reasons, participants need to download speed data and generate edge labels from them locally. We provide the script to do that and it will take approx. 45 minutes for cc generation and approx. 90 minutes for eta generation. Node data can be downloaded from AWS from the public links below.
<img src="./img/data_overview.svg">

More details can be found in `data_pipeline/README.md`


### Folder Structure

Unzip:
```bash
BASEDIR=data
unzip LONDON_2022.zip -d $BASEDIR
```
After unzipping, you should have received the following merged folder structure

```
├── bounding_boxes.geojson
├── loop_counter
│   ├── london
│   │   └── counters_daily_by_node.parquet
│   ├── madrid
│   │   └── counters_daily_by_node.parquet
│   └── melbourne
│       └── counters_daily_by_node.parquet
├── road_graph
│   ├── london
│   │   ├── cell_mapping.parquet
│   │   ├── road_graph_edges.parquet
│   │   ├── road_graph_nodes.parquet
│   │   └── road_graph_supersegments.parquet
│   ├── madrid
│   │   ├── cell_mapping.parquet
│   │   ├── road_graph_edges.parquet
│   │   ├── road_graph_nodes.parquet
│   │   └── road_graph_supersegments.parquet
│   └── melbourne
│       ├── cell_mapping.parquet
│       ├── road_graph_edges.parquet
│       ├── road_graph_nodes.parquet
│       └── road_graph_supersegments.parquet
├── movie
│   ├── london
|   │   ├── 2019-07-01_london_8ch.h5
|   │      ...
│   ├── madrid
|   │   ├── 2019-07-01_madrid_8ch.h5
|   │      ...
│   └── melbourne
|   │   ├── 2019-07-01_melbourne_8ch.h5
|   │      ...
├── speed_classes
│   ├── london
|   │   ├── speed_classes_2019-07-01.parquet
|   │      ...
│   ├── madrid
|   │   ├── speed_classes_2019-07-01.parquet
|   │      ...
│   └── melbourne
|   │   ├── speed_classes_2019-07-01.parquet
|   │      ...
├── submissions
│   ├── london
│   ├── madrid
│   └── melbourne
├── test
│   ├── london
│   ├── madrid
│   └── melbourne
└── train
    ├── london
    │   ├── input
    │   │   ├── counters_2019-07-01.parquet
    │   │       ...
    │   └── labels
    ├── madrid
    │   ├── input
    │   │   ├── counters_2021-06-01.parquet
    │   │       ...
    │   └── labels
    └── melbourne
        ├── input
        │   ├── counters_2020-06-01.parquet
        │       ...
        └── labels
```

**bounding_boxes.geojson**

This file provides the bounding boxes for the competition data in the three
cities London, Madrid and Melbourne.


**loop_counter**

This folder contains open traffic count data for London, Madrid and Melbourne.
The locations of the loop counters have been mapped to the simplified OSM road graph (see below).

London - Powered by TfL Open Data
The raw TIMS detector data is available from https://roads.data.tfl.gov.uk/
The license is based on version 2.0 of the Open Government Licence allowing to copy, publish,
transmit and adapt the information in commercial and non-commercial applications, see
https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service

London - Highways England:
The data is available through the API at https://webtris.highwaysengland.co.uk/
Licensed under Open Government Licence allowing to copy, publish, transmit and adapt the
information in commercial and non-commercial applications, see
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

Madrid - Madrid City Council
The data is available on the Madrid open data portal at
https://datos.madrid.es/egob/catalogo/208627-0-transporte-ptomedida-historico and
https://datos.madrid.es/egob/catalogo/202468-0-intensidad-trafico
The license terms allow the reuse of the data for commercial and non-commercial purposes,
see https://datos.madrid.es/egob/catalogo/aviso-legal

Melbourne - Department of Transport, Victoria State Government
The data is available through the open data portal of the State of Victoria at
https://discover.data.vic.gov.au/dataset/traffic-lights and
https://discover.data.vic.gov.au/dataset/traffic-signal-volume-data
Licensed under Creative Commons Attribution 4.0 International allowing the share and adapt
in any form, see https://creativecommons.org/licenses/by/4.0/


**road_graph**

© OpenStreetMap contributors - this folder contains a simplified road graph for
London, Madrid and Melbourne based on Map data from OpenStreetMap.

OpenStreetMap (OSM) data licensed under the Open Data Commons Open Database License (ODbL)
by the OpenStreetMap Foundation (OSMF):
"You are free to copy, distribute, transmit and adapt our data, as long as you credit
OpenStreetMap and its contributors. If you alter or build upon our data, you may distribute
the result only under the same licence."
See details of the copyright at https://www.openstreetmap.org/copyright and the full legal
code of the license at https://opendatacommons.org/licenses/odbl/1-0/

The files `road_graph_edges.parquet` and `road_graph_nodes.parquet` contain the simplified graph
with the bare minimum attributes as node/edge graph structure. Where necessary additional
nodes were added for the mapping to the loop counter locations (see above). Otherwise the
the original OSM node IDs were retained.

The file `cell_mapping.parquet` provides a mapping of each edge in the road graph to the
intersecting cells in the Traffic4cast Traffic Map Movie grid for the corresponding city
(based on `bounding_boxes.geojson`).

The file `road_graph_supersegments.parquet` contains the list of supersegments for the extended competition along with the list of corresponding nodes.


**submissions**

This folder is empty and will be used by participants for preparing their submissions.


**test**

This folder contains the test inputs for generating a submission,
for more information see https://github.com/iarai/NeurIPS2022-traffic4cast

**movie**
The dynamic traffic map movie data is provided by HERE and can be downloaded from https://developer.here.com/sample-data.
This data needs only be downloaded if participants want to re-write their own mapping to the road graph.

The data for one day is given as a (288,495,436,8) tensor
in a HDF5 file (.h5 suffix). The first 3 dimensions encode the 5-min time bin, height
and width of each "image". The first two of the 8 channels encode the aggregated volume and average speed of all underlying probes whose heading is between 0 and
90 degrees (i.e. NE), the next two the same aggregation of volume and speed for all
probes heading NW, the following two for SE and SW, respectively.


**speed_classes**
The dynamic speed classifications are derived from the traffic map movie data provided by HERE and can be downloaded from https://developer.here.com/sample-data.

Values per 15-minute time bins are given as a columnar table in a Parquet file
(.parquet suffix). The entries in this table can be joined with a fully connected road
graph using the tools provided at https://github.com/iarai/NeurIPS2022-traffic4cast.

**train**

This folder provides the training input data for all three cities.

The subfolder <city>/input contains the loop_counter (see above) information pre-processed
in order to be directly consumable by a data loader.


The subfolder <city>/labels is being generated after downloading the Traffic4cast 2022 datasets
from https://developer.here.com/sample-data and running the label pre-processing script,
for more information see https://github.com/iarai/NeurIPS2022-traffic4cast





For each city, we're using approx. 6 months of data, interleaving training and test data on a weekly basis (1 week of training and 1 week for test):
| City          | Period            |
|---------------|-------------------|
| London        | 2019-07-01 - 2020-01-31 |
| Madrid        | 2021-06-01 - 2021-12-31 |
| Melbourne     | 2020-06-01 - 2020-12-30 |

More information can be found:
* [data pipeline](data_pipline/README.md)
* [data specification](README_DATA_SPECIFICATION.md)


### Generating Training Labels
The script `t4c22/prepare_training_data_cc.py`/`t4c22/prepare_training_data_eta.py` prepares congestion class/supersemgent eta training labels using speed_classes. This should take less than 45 minutes for cc generation and less than 90 minutes for eta generation.

Prior to starting you should have downloaded and extracted the following 5
ZIP files to the same working data directory (`<working_dir>`):
- LONDON_2022.zip from https://developer.here.com/sample-data
- MADRID_2022.zip from https://developer.here.com/sample-data
- MELBOURNE_2022.zip from https://developer.here.com/sample-data
- T4C_INPUTS_2022.zip from https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip
- T4C_INPUTS_ETA_2022.zip from https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip



These scripts require the following folders in `<working_dir>`
```
├── road_graph
│   ├── london
│   ├── madrid
│   └── melbourne
├── speed_classes
│   ├── london
│   ├── madrid
│   └── melbourne
└── train
    ├── london
    ├── madrid
    └── melbourne
```
They will create the following label files
```
└── train
    ├── london
    │   └── labels
    │       ├── cc_labels_2019-07-01.parquet
    │       ├── eta_labels_2019-07-01.parquet
    │          ...
    ├── madrid
    │   └── labels
    │       ├── cc_labels_2021-06-01.parquet
    │       ├── eta_labels_2021-06-01.parquet
    │          ...
    └── melbourne
        └── labels
            ├── cc_labels_2020-06-01.parquet
            ├── eta_labels_2020-06-01.parquet
                ...
```

They have the following command-line interface:
```
Usage:
  prepare_training_data_cc.py [-h] -d DATA_FOLDER [-r]
  prepare_training_data_eta.py [-h] -d DATA_FOLDER [-r]

Arguments:
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Folder containing T4c data
  -r, --resume          Resume processing without regenerating existing files
```



## Metric
### Metric core competition (congestion classes `cc`)
We use masked cross-entropy loss on congestion classes:



$$\ell(\hat{y}, y) = \sum_{n=1}^N \frac{1}{\Sigma_{n=1}^N   w_{y_n} \cdot \mathbb{1}\\{y_n \not= \text{ignore}\\\_\text{index}\\}} l_n, \quad
          l_n = - w_{y_n} \log \frac{\hat{y}\_{n,y_n}+\varepsilon}{\Sigma\_{c=0}^{C-1} \hat{y}_{n,c}+\varepsilon} \cdot \mathbb{1}\\{y_n \not= \text{ignore}\\\_\text{index}\\}$$


where
* $\hat{y} \in \mathbb{R}^{N \times C}$ is the input (probabilities, if logits are used, then replace $x_{n,c}$ and $y_n$ by $\exp(x_{n,c})$ and $\exp(y_{n})$ ),
* $y \in \\{0,...,C-1\\}^N$ is the target,
* $w \in \mathbb{R}^C$ is the class weight,
* $C \in \mathbb{N}$ is the number of classes,
* $N \in \mathbb{N}$ is the number of samples,
* $\varepsilon \in \mathbb{R}^+$ is small constant preventing overflow in $\log(0)$,
* $\text{ignore}\\\_\text{index} \in C \cup \\{\bot\\}$ specifies a target value that is ignored or none ( $\bot$ ).

Cross-entropy loss will penalize small predicted probabilities disproportionately.

In our setting,
* $C=4$ and $\text{ignore}\\\_\text{index}=0$, i.e., we always **mask** on unclassified edges in ground truth
  * 0 = unclassified
  * 1 = green (uncongested)
  * 2 = yellow (slowed down)
  * 3 = red (congest)
* $N$ goes over edges and timestamps
* $w$ macro averaging: $w_c = \frac{N}{|C| \cdot \Sigma_{i=1}^N{   \mathbb{1} \\{y_i = c\\} }}$ for $c \in C$ simply calculates the mean for each ground truth class, giving equal weight to each class ([scikit multiclas classification](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification)). In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance.  In our case, since we have more red than yellow than green in all cities and since we're interested in capturing congested situations well, we take this approach.

We provide
* the weights $w$ for each city
* average the macro-averaged city scores of the 3 cities to get an overall score.


References: [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [scikit multiclass classification](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification)

### Metric extended competition (supersegment travel times `eta`)
We use [L1 Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) for the extended competitions. Notice that in contrast to the core competitions, there is no "evaluation mask", i.e. all supersegments have an ETA label.

## Test set

### Test set core competition (congestion classification `cc`)
We will restrict evaluation to a subset 6am-10pm (including weekends).
However, all data can be used for training.

We are providing the code we use for evaluation and code to generate own validation sets.

### Test set extended competition (supersegment travel times `eta`)
We use the exactly same test set for the extended competition as for the core competition (same input files `test/<city>/input/counters_test.parquet`)

## Data loaders
We provide a plain torch and torch geometric data loader, see `t4c22.dataloading` and the working examples under `exploration`.


## Baselines (forthcoming)
The tasks can be tackled in many ways, from GNN to attention-based to working on the grid (movies).

In the mean-time, see the examples under `exploration` with working data loading.



## Cite
Please cite this repo along with the papers of the previous competitions:

```
@InProceedings{pmlr-v176-eichenberger22a,
  title = 	 {Traffic4cast at NeurIPS 2021 - Temporal and Spatial Few-Shot Transfer Learning in Gridded Geo-Spatial Processes},
  author =       {Eichenberger, Christian and Neun, Moritz and Martin, Henry and Herruzo, Pedro and Spanring, Markus and Lu, Yichao and Choi, Sungbin and Konyakhin, Vsevolod and Lukashina, Nina and Shpilman, Aleksei and Wiedemann, Nina and Raubal, Martin and Wang, Bo and Vu, Hai L. and Mohajerpoor, Reza and Cai, Chen and Kim, Inhi and Hermes, Luca and Melnik, Andrew and Velioglu, Riza and Vieth, Markus and Schilling, Malte and Bojesomo, Alabi and Marzouqi, Hasan Al and Liatsis, Panos and Santokhi, Jay and Hillier, Dylan and Yang, Yiming and Sarwar, Joned and Jordan, Anna and Hewage, Emil and Jonietz, David and Tang, Fei and Gruca, Aleksandra and Kopp, Michael and Kreil, David and Hochreiter, Sepp},
  booktitle = 	 {Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track},
  pages = 	 {97--112},
  year = 	 {2022},
  editor = 	 {Kiela, Douwe and Ciccone, Marco and Caputo, Barbara},
  volume = 	 {176},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--14 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v176/eichenberger22a/eichenberger22a.pdf},
  url = 	 {https://proceedings.mlr.press/v176/eichenberger22a.html},
  abstract = 	 {The IARAI Traffic4cast competitions at NeurIPS 2019 and 2020 showed that neural networks can successfully predict future traffic conditions 1 hour into the future on simply aggregated GPS probe data in time and space bins. We thus reinterpreted the challenge of forecasting traffic conditions as a movie completion task. U-Nets proved to be the winning architecture, demonstrating an ability to extract relevant features in this complex real-world geo-spatial process. Building on the previous competitions, Traffic4cast 2021 now focuses on the question of model robustness and generalizability across time and space. Moving from one city to an entirely different city, or moving from pre-COVID times to times after COVID hit the world thus introduces a clear domain shift. We thus, for the first time, release data featuring such domain shifts. The competition now covers ten cities over 2 years, providing data compiled from over $10^{12}$ GPS probe data. Winning solutions captured traffic dynamics sufficiently well to even cope with these complex domain shifts. Surprisingly, this seemed to require only the previous 1h traffic dynamic history and static road graph as input. }
}

@InProceedings{pmlr-v133-kopp21a,
  title =      {Traffic4cast at NeurIPS 2020 - yet more on the unreasonable effectiveness of gridded geo-spatial processes},
  author =       {Kopp, Michael and Kreil, David and Neun, Moritz and Jonietz, David and Martin, Henry and Herruzo, Pedro and Gruca, Aleksandra and Soleymani, Ali and Wu, Fanyou and Liu, Yang and Xu, Jingwei and Zhang, Jianjin and Santokhi, Jay and Bojesomo, Alabi and Marzouqi, Hasan Al and Liatsis, Panos and Kwok, Pak Hay and Qi, Qi and Hochreiter, Sepp},
  booktitle =      {Proceedings of the NeurIPS 2020 Competition and Demonstration Track},
  pages =      {325--343},
  year =      {2021},
  editor =      {Escalante, Hugo Jair and Hofmann, Katja},
  volume =      {133},
  series =      {Proceedings of Machine Learning Research},
  month =      {06--12 Dec},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v133/kopp21a/kopp21a.pdf},
  url =      {https://proceedings.mlr.press/v133/kopp21a.html},
  abstract =      {The IARAI Traffic4cast competition at NeurIPS 2019 showed that neural networks can successfully predict future traffic conditions 15 minutes into the future on simply aggregated GPS probe data  in time and space bins, thus interpreting the challenge of forecasting traffic conditions as a movie completion task. U-nets proved to be the winning architecture then, demonstrating an ability  to extract relevant features in the complex, real-world, geo-spatial process that is traffic derived from a large data set. The IARAI Traffic4cast challenge at NeurIPS 2020 build on the insights of the previous year and sought to both challenge some assumptions inherent in our 2019 competition design and explore how far this neural network technique can be pushed. We found that the  prediction horizon can be extended successfully to 60 minutes into the future, that there is further evidence that traffic depends more on recent dynamics than on the additional static or dynamic location specific data provided and that a reasonable starting point when exploring a general aggregated geo-spatial process in time and space is a U-net architecture.}
}

@InProceedings{pmlr-v123-kreil20a,
  title =      {The surprising efficiency of framing geo-spatial time series forecasting as a video prediction task – Insights from the IARAI \t4c Competition at NeurIPS 2019},
  author =       {Kreil, David P and Kopp, Michael K and Jonietz, David and Neun, Moritz and Gruca, Aleksandra and Herruzo, Pedro and Martin, Henry and Soleymani, Ali and Hochreiter, Sepp},
  booktitle =      {Proceedings of the NeurIPS 2019 Competition and Demonstration Track},
  pages =      {232--241},
  year =      {2020},
  editor =      {Escalante, Hugo Jair and Hadsell, Raia},
  volume =      {123},
  series =      {Proceedings of Machine Learning Research},
  month =      {08--14 Dec},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v123/kreil20a/kreil20a.pdf},
  url =      {https://proceedings.mlr.press/v123/kreil20a.html},
  abstract =      {Deep Neural Networks models are state-of-the-art solutions in accurately forecasting future video frames in a movie.  A successful video prediction model needs to extract and encode semantic features that describe the complex spatio-temporal correlations within image sequences of the real world.  The IARAI Traffic4cast Challenge of the NeurIPS Competition Track 2019 for the first time introduced the novel argument that this is also highly relevant for urban traffic. By framing traffic prediction as a movie completion task, the challenge requires models to take advantage of complex geo-spatial and temporal patterns of the underlying process. We here report on the success and insights obtained in a first Traffic Map Movie forecasting challenge. Although short-term traffic prediction is considered hard, this novel approach allowed several research groups to successfully predict future traffic states in a purely data-driven manner from pixel space. We here expand on the original rationale, summarize key findings, and discuss promising future directions of the t4c competition at NeurIPS.}
}
```

* [Traffic4cast at NeurIPS 2021 - Temporal and Spatial Few-Shot Transfer Learning in Gridded Geo-Spatial Processes](https://proceedings.mlr.press/v176/eichenberger22a/eichenberger22a.pdf)
* [Traffic4cast at NeurIPS 2020 - yet more on the unreasonable effectiveness of gridded geo-spatial processes](http://proceedings.mlr.press/v133/kopp21a/kopp21a.pdf)
* [The surprising efficiency of framing geo-spatial time series forecasting as a video prediction task – Insights from the IARAI \t4c Competition at NeurIPS 2019](http://proceedings.mlr.press/v123/kreil20a/kreil20a.pdf)
