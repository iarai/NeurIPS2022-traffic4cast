#  Traffic4cast 2022 Competition: from few public vehicle counters to entire city-wide traffic

![traffic4cast2020](t4c20logo.png)

* Predict congestion classes (red/yellow/green) and average speed.
* Work with real world data from industrial-sized fleets and vehicle counters.
* Apply Graph Learning to predict traffic dynamics on the road graph edges given sparse node data.


## About
[This repo](https://github.com/iarai/NeurIPS2022-traffic4cast) will contain all information about the Traffic4cast 2021 NeurIPS competition for participants. It will contain detailed information about the competition and data as well as code and issue tracker.

## Introduction
Our Traffic4cast competition series at NeurIPS has contributed both methodological and practical insights to advance the application of AI to forecasting traffic and other spatial processes. Now, going beyond the Traffic4cast challenges at NeurIPS 2019, 2020, and 2021, this year we will explore models that have the ability to **generalize loosely related temporal vertex data on just a few nodes** to **predict dynamic future traffic states on the edges of the entire road graph**. Specifically, in our core challenge we invite participants to predict for three cities the **congestion classes** known from the red, yellow, or green colouring of roads on a common traffic map for the entire road graph 15min into the future. We provide **car count data from spatially sparse vehicle counters** in these three cities in 15min aggregated time bins for one hour prior to the prediction time slot. For our extended challenge participants are asked to predict the **actual average speeds** on each road segment in the graph 15min into the future.


![traffic4cast2022](Main-image-web-08.png)

Interested in understanding traffic around the globe in a novel way? [Join us](https://www.iarai.ac.at/traffic4cast/forums/) to help shape this year’s competition.


## Schedule
* Beginning of July 2022: Traffic data available
* 15th of October 2022: Submission to Traffic4cast 2022 leaderboard deadline
* 21st of October 2022: Abstract submission deadline

## Cite
Please cite the papers of the previous competitions:

```
@misc{https://doi.org/10.48550/arxiv.2203.17070,
  doi = {10.48550/ARXIV.2203.17070},
  url = {https://arxiv.org/abs/2203.17070},
  author = {Eichenberger, Christian and Neun, Moritz and Martin, Henry and Herruzo, Pedro and Spanring, Markus and Lu, Yichao and Choi, Sungbin and Konyakhin, Vsevolod and Lukashina, Nina and Shpilman, Aleksei and Wiedemann, Nina and Raubal, Martin and Wang, Bo and Vu, Hai L. and Mohajerpoor, Reza and Cai, Chen and Kim, Inhi and Hermes, Luca and Melnik, Andrew and Velioglu, Riza and Vieth, Markus and Schilling, Malte and Bojesomo, Alabi and Marzouqi, Hasan Al and Liatsis, Panos and Santokhi, Jay and Hillier, Dylan and Yang, Yiming and Sarwar, Joned and Jordan, Anna and Hewage, Emil and Jonietz, David and Tang, Fei and Gruca, Aleksandra and Kopp, Michael and Kreil, David and Hochreiter, Sepp},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Traffic4cast at NeurIPS 2021 - Temporal and Spatial Few-Shot Transfer Learning in Gridded Geo-Spatial Processes},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
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
