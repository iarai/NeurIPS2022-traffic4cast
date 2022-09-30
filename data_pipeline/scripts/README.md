Directory Layout for
* `download_all.sh` downloads to `here_downloads`
* `extract_all.sh` extracts from `here_downloads` to `here_extracted` and links `<year>/movie` to a subfolder of `here_extracted`
* `run_all.sh` works under `<year>/movie` as data dir for the pipeline

```
<DATA_ROOT>
├── 2020
│   └── movie
│       ├── berlin -> data/here_extracted/2020/BERLIN/training
│       ├── istanbul -> data/here_extracted/2020/ISTANBUL/training
│       └── moscow -> data/here_extracted/2020/MOSCOW/training
├── 2021
│   ├── movie
│   │   ├── antwerp -> data/here_extracted/2021/ANTWERP/training
│   │   ├── bangkok -> data/here_extracted/2021/BANGKOK/training
│   │   ├── barcelona -> data/here_extracted/2021/BARCELONA/training
│   │   ├── berlin -> data/here_extracted/2021/BERLIN/training
│   │   ├── chicago -> data/here_extracted/2021/CHICAGO/training
│   │   ├── istanbul -> data/here_extracted/2021/ISTANBUL/training
│   │   ├── melbourne -> data/here_extracted/2021/MELBOURNE/training
│   │   └── moscow -> data/here_extracted/2021/MOSCOW/training
├── 2022
│   └── movie
│       ├── berlin -> data/here_extracted/2022/movie/berlin
│       ├── london -> data/here_extracted/2022/movie/london
│       ├── madrid -> data/here_extracted/2022/movie/madrid
│       └── melbourne -> data/here_extracted/2022/movie/melbourne
├── here_downloads
│   ├── 2019
│   │   ├── Berlin_2018.zip
│   │   ├── Istanbul_2018.zip
│   │   └── Moscow_2018.zip
│   ├── 2020
│   │   ├── BERLIN.tar
│   │   ├── ISTANBUL.tar
│   │   └── MOSCOW.tar
│   ├── 2021
│   │   ├── ANTWERP.tar.gz
│   │   ├── BANGKOK.tar.gz
│   │   ├── BARCELONA.tar.gz
│   │   ├── BERLIN.tar.gz
│   │   ├── CHICAGO.tar.gz
│   │   ├── ISTANBUL.tar.gz
│   │   ├── MELBOURNE.tar.gz
│   │   ├── MOSCOW.tar.gz
│   │   ├── NEWYORK.tar.gz
│   │   └── VIENNA.tar.gz
│   └── 2022
│       ├── LONDON_2022.zip
│       ├── MADRID_2022.zip
│       └── MELBOURNE_2022.zip
└── here_extracted
    ├── 2020
    │   ├── BERLIN
    │   │   ├── BERLIN_static_2019.h5
    │   │   ├── testing
    │   │   │   ├── 2019-07-02_test.h5
    │   │   │   ├── ...
    │   │   ├── training
    │   │   │   ├── 2019-01-01_berlin_9ch.h5
    │   │   │   ├── ...
    │   │   └── validation
    │   │       ├── 2019-07-01_berlin_9ch.h5
    │   │       ├── ...
    │   ├── ISTANBUL
    │   │   ├── ISTANBUL_static_2019.h5
    │   │   ├── testing
    │   │   │   ├── 2019-07-02_test.h5
    │   │   │   ├── ...
    │   │   ├── training
    │   │   │   ├── 2019-01-01_istanbul_9ch.h5
    │   │   │   ├── ...
    │   │   └── validation
    │   │       ├── 2019-07-01_istanbul_9ch.h5
    │   │       ├── ...
    │   └── MOSCOW
    │       ├── MOSCOW_static_2019.h5
    │       ├── testing
    │       │   ├── 2019-07-02_test.h5
    │       │   ├── ...
    │       ├── training
    │       │   ├── 2019-01-01_moscow_9ch.h5
    │       │   ├── ...
    │       └── validation
    │           ├── 2019-07-01_moscow_9ch.h5
    │           ├── ...
    ├── 2021
    │   ├── ANTWERP
    │   │   ├── ANTWERP_map_high_res.h5
    │   │   ├── ANTWERP_static.h5
    │   │   └── training
    │   │       ├── 2019-01-02_ANTWERP_8ch.h5
    │   │       ├── ...
    │   ├── BANGKOK
    │   │   ├── BANGKOK_map_high_res.h5
    │   │   ├── BANGKOK_static.h5
    │   │   └── training
    │   │       ├── 2019-01-02_BANGKOK_8ch.h5
    │   │       ├── ...
    │   ├── BARCELONA
    │   │   ├── BARCELONA_map_high_res.h5
    │   │   ├── BARCELONA_static.h5
    │   │   └── training
    │   │       ├── 2019-01-02_BARCELONA_8ch.h5
    │   │       ├── ...
    │   ├── BERLIN
    │   │   ├── BERLIN_map_high_res.h5
    │   │   ├── BERLIN_static.h5
    │   │   ├── BERLIN_test_additional_temporal.h5
    │   │   ├── BERLIN_test_temporal.h5
    │   │   └── training
    │   │       ├── 2019-01-02_BERLIN_8ch.h5
    │   │       ├── ...
    │   ├── CHICAGO
    │   │   ├── CHICAGO_map_high_res.h5
    │   │   ├── CHICAGO_static.h5
    │   │   ├── CHICAGO_test_additional_temporal.h5
    │   │   ├── CHICAGO_test_temporal.h5
    │   │   └── training
    │   │       ├── 2019-01-02_CHICAGO_8ch.h5
    │   │       ├── ...
    │   ├── ISTANBUL
    │   │   ├── ISTANBUL_map_high_res.h5
    │   │   ├── ISTANBUL_static.h5
    │   │   ├── ISTANBUL_test_additional_temporal.h5
    │   │   ├── ISTANBUL_test_temporal.h5
    │   │   └── training
    │   │       ├── 2019-01-02_ISTANBUL_8ch.h5
    │   │       ├── ...
    │   ├── MELBOURNE
    │   │   ├── MELBOURNE_map_high_res.h5
    │   │   ├── MELBOURNE_static.h5
    │   │   ├── MELBOURNE_test_additional_temporal.h5
    │   │   ├── MELBOURNE_test_temporal.h5
    │   │   └── training
    │   │       ├── 2019-01-02_MELBOURNE_8ch.h5
    │   │       ├── ...
    │   ├── MOSCOW
    │   │   ├── MOSCOW_map_high_res.h5
    │   │   ├── MOSCOW_static.h5
    │   │   └── training
    │   │       ├── 2019-01-02_MOSCOW_8ch.h5
    │   │       ├── ...
    │   ├── NEWYORK
    │   │   ├── NEWYORK_map_high_res.h5
    │   │   ├── NEWYORK_static.h5
    │   │   ├── NEWYORK_test_additional_spatiotemporal.h5
    │   │   └── NEWYORK_test_spatiotemporal.h5
    │   └── VIENNA
    │       ├── VIENNA_map_high_res.h5
    │       ├── VIENNA_static.h5
    │       ├── VIENNA_test_additional_spatiotemporal.h5
    │       └── VIENNA_test_spatiotemporal.h5
    └── 2022
        └── movie
            ├── london
            │   ├── 2019-07-01_london_8ch.h5
            │   ├── ...
            ├── madrid
            │   ├── 2021-06-01_madrid_8ch.h5
            │   ├── ...
            └── melbourne
                ├── 2020-06-01_melbourne_8ch.h5
                ├── ...



```
