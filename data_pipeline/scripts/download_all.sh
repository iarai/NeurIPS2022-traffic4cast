#
# Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

export DATA_ROOT="YOUR_LOCATION_SHOULD_GO_HERE"
# get it from https://developer.here.com/sample-data
export BASE_URL=

# download -> here_downloads/YEAR

mkdir -p ${DATA_ROOT}/here_downloads/2019
wget ${BASE_URL}/2019/Berlin_2018.zip -P ${DATA_ROOT}/here_downloads/2019
wget ${BASE_URL}/2019/Istanbul_2018.zip -P ${DATA_ROOT}/here_downloads/2019
wget ${BASE_URL}/2019/Moscow_2018.zip -P ${DATA_ROOT}/here_downloads/2019

mkdir -p ${DATA_ROOT}/here_downloads/2020
wget ${BASE_URL}/2020/BERLIN.tar -P ${DATA_ROOT}/here_downloads/2020
wget ${BASE_URL}/2020/ISTANBUL.tar -P ${DATA_ROOT}/here_downloads/2020
wget ${BASE_URL}/2020/MOSCOW.tar -P ${DATA_ROOT}/here_downloads/2020

mkdir -p ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/ANTWERP.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/BANGKOK.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/BARCELONA.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/BERLIN.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/CHICAGO.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/ISTANBUL.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/MELBOURNE.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/ANTWERP.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/MOSCOW.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/NEWYORK.tar.gz -P ${DATA_ROOT}/here_downloads/2021
wget ${BASE_URL}/2021/VIENNA.tar.gz -P ${DATA_ROOT}/here_downloads/2021

mkdir -p ${DATA_ROOT}/here_downloads/2022
wget ${BASE_URL}/2022/LONDON_2022.zip -P ${DATA_ROOT}/here_downloads/2022
wget ${BASE_URL}/2022/MADRID_2022.zip -P ${DATA_ROOT}/here_downloads/2022
wget ${BASE_URL}/2022/MELBOURNE_2022.zip -P ${DATA_ROOT}/here_downloads/2022
