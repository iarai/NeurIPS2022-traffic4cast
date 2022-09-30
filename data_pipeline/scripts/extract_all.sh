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

# extract here_downloads/YEAR -> here_extracted/YEAR
mkdir -p ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/BERLIN.tar -C ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/ISTANBUL.tar -C ${DATA_ROOT}/here_extracted/2020
tar xf ${DATA_ROOT}/here_downloads/2020/MOSCOW.tar -C ${DATA_ROOT}/here_extracted/2020

mkdir -p ${DATA_ROOT}/2020/movie
ln -s ${DATA_ROOT}/here_extracted/2020/BERLIN/training ${DATA_ROOT}/2020/movie/berlin
ln -s ${DATA_ROOT}/here_extracted/2020/ISTANBUL/training ${DATA_ROOT}/2020/movie/istanbul
ln -s ${DATA_ROOT}/here_extracted/2020/MOSCOW/training ${DATA_ROOT}/2020/movie/moscow

mkdir -p ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/ANTWERP.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BANGKOK.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BARCELONA.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/BERLIN.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/CHICAGO.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/ISTANBUL.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/MELBOURNE.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/MOSCOW.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/NEWYORK.tar.gz -C ${DATA_ROOT}/here_extracted/2021
tar zxf ${DATA_ROOT}/here_downloads/2021/VIENNA.tar.gz -C ${DATA_ROOT}/here_extracted/2021

mkdir -p ${DATA_ROOT}/2021/movie
ln -s ${DATA_ROOT}/here_extracted/2021/ANTWERP/training ${DATA_ROOT}/2021/movie/antwerp
ln -s ${DATA_ROOT}/here_extracted/2021/BANGKOK/training ${DATA_ROOT}/2021/movie/bangkok
ln -s ${DATA_ROOT}/here_extracted/2021/BARCELONA/training ${DATA_ROOT}/2021/movie/barcelona
ln -s ${DATA_ROOT}/here_extracted/2021/BERLIN/training ${DATA_ROOT}/2021/movie/berlin
ln -s ${DATA_ROOT}/here_extracted/2021/CHICAGO/training ${DATA_ROOT}/2021/movie/chicago
ln -s ${DATA_ROOT}/here_extracted/2021/ISTANBUL/training ${DATA_ROOT}/2021/movie/istanbul
ln -s ${DATA_ROOT}/here_extracted/2021/MELBOURNE/training ${DATA_ROOT}/2021/movie/melbourne
ln -s ${DATA_ROOT}/here_extracted/2021/MOSCOW/training ${DATA_ROOT}/2021/movie/moscow

mkdir -p ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/LONDON_2022.zip -d ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/MADRID_2022.zip -d ${DATA_ROOT}/here_extracted/2022/
unzip ${DATA_ROOT}/here_downloads/2022/MELBOURNE_2022.zip -d ${DATA_ROOT}/here_extracted/2022/

mkdir -p ${DATA_ROOT}/2022/movie
ln -s ${DATA_ROOT}/here_extracted/2022/movie/london ${DATA_ROOT}/2022/movie/london
ln -s ${DATA_ROOT}/here_extracted/2022/movie/madrid ${DATA_ROOT}/2022/movie/madrid
ln -s ${DATA_ROOT}/here_extracted/2022/movie/melbourne ${DATA_ROOT}/2022/movie/melbourne
