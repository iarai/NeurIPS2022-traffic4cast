# Traffic4cast 2022 Data Pipeline

We will publish the code to generate the speed classifications from the traffic movies later in this sub-folder.

### Color Scheme
<img src="../img/data_pipeline_color_scheme.svg">

### Symbols Used
<img src="../img/data_pipeline_symbols.svg">


## Data Pipeline Node Data: Vehicle Counts
<img src="../img/data_pipeline_node_data.svg">



| Item               | Description                |
|--------------------|----------------------------|
| download and preprocessing | Download the public data, purge and transform in intermediate common data format. |
| postprocessing             | From the normalized loop counter data, join with OSM IDs. |
| Grouping (+Sampling)       | Group 4 15-minute counts for input. Sample test data.      |

### Loop Counter Data Sources

| City | Provider | Counter Locations Dataset  | Historical Counts Dataset | License |
|------|----------|----------------------------|---------------------------|----------|
| Madrid | city council (Ayuntamiento de Madrid) | [Tráfico. Ubicación de los puntos de medida del tráfico](https://datos.madrid.es/egob/catalogo/202468-0-intensidad-trafico) | [Tráfico. Histórico de datos del tráfico desde 2013](https://datos.madrid.es/egob/catalogo/208627-0-transporte-ptomedida-historico) | [Aviso Legal](https://datos.madrid.es/egob/catalogo/aviso-legal) |
| Melbourne | Department of Transport in the State of Victoria |  [Traffic Lights](https://discover.data.vic.gov.au/dataset/traffic-lights) | [Signal Volume Data](https://discover.data.vic.gov.au/dataset/traffic-signal-volume-data) | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |
| London  | TfL  | [TIMS detector data](https://roads.data.tfl.gov.uk/) | [TIMS detector data](https://roads.data.tfl.gov.uk/) | [Terms and Conditions Transport Data Service](https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service) |
| London  | Highways England |[Highways England network journey time and traffic flow data](https://data.gov.uk/dataset/9562c512-4a0b-45ee-b6ad-afc0f99b841f/highways-england-network-journey-time-and-traffic-flow-data) | [Traffic Flow data - Sites and Historical Reports](https://webtris.highwaysengland.co.uk/)|  [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |



## Data Pipeline Edge Data: Congestion Classes from GPS Speeds
<img src="../img/data_pipeline_edge_data.svg">


| Item               | Description                |
|--------------------|----------------------------|
| Road Selection     | Select OSM primary features of type "highway" in the bounding box, introduce nodes for loop counters and simplify the road graph (remove dead-ends and unconnected components etc.). |
| Spatial Join       | Intersect OSM roads geometries with map movie cell geometries. |
| Combination        | Combine Traffic Map values with OSM IDs. |
| Generate CC Labels | Generate congestion classes (green=1, yellow=2, red=3) from the current segment medium speeds, the free flow speeds computed for the segment from the traffic map movies and the OSM signalled speeds. If no or not enough dynamic speed data is available, do not classify (unclassified=0). |
