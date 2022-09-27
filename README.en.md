1st place in AIJ-2020 https://github.com/ai-forever/no_flood_with_ai_aij2020

# Model for forecasting the water level of Amur River (AIJ-2020)
<p align="center">
  <img src="pics/NN - прогноз на 5 лет.png" width="100%">
</p>

Repository structure:
* EDA.ipynb - jupyter notebook with data research and comments
* amurlevel_model - main model of repository
* predict.py - script for 10-days forecasting the water level
* train.py - script for model training for 10-days forecasting the water level
* Dockerfile_train - Dockerfile for training
* Dockerfile_predict - Dockerfile for inference
* data - directory with necessary data for traning and additional data needed for model, а также дополнительными данными, необходимыми для работы модели (model weights)
* data/mean_std_stats.json - Calculated stats with mean,std for numerical features
* data/weights-aij2020amurlevel-2017.h5 - model weights obtained with all available data till 2017-12-31
* data/weights-aij2020amurlevel-2012.h5 - model weights obtained with all available data till 2012-12-03
* abstracts.pdf - short presentation of model (russian)

### Data repository structure
Data consists of all historical data from hydroposts and hydrostations and meteo data in AISORI format. Data with snow cover and ice (_ice.csv) or data with daily discharge (_disch_d.csv) are not required - if there is no data for some hydrostation, this data will be skipped.
**WARNING! The input must be in exactly the same format as the original data from the organizers (the same number of fields in the same order), several examples are attached in the repository. You can download data [here](https://storage.yandexcloud.net/datasouls-ods/materials/c8b9bab3/datasets.zip)**
```
data/
---hydro/
------05001_daily.csv
------05001_ice.csv
------05002_daily.csv
------...
---meteo_new/
------4443141.csv
------4483311.csv
------...
---mean_std_stats.json
---weights-aij2020amurlevel-2017.h5
---...
```

### Weather forecasting models
There are two weather forecasting services:
<ol>
  <li> Yandex.weather (7 daus forecasting, 5000 requests/day, 30 days free trial) - https://yandex.ru/dev/weather/doc/dg/concepts/pricing.html/ </li>
  <li> Openweather (16 days forecasting, 5000 requests/month, 10$/month) - https://rapidapi.com/community/api/open-weather-map?endpoint=53aa6042e4b051a76d241b79 </li>
</ol>

You must set correct credentials in config.py. 

### Model settings
amurlevel_model/config.py - model settings. The most important parameters are described below:
* DATASETS_PATH - absolute path with input data
* DAYS_FORECAST - forecasting period in days (default 10 days)
* ALL_STATIONS - ID's of all hydrostations and hydroposts which will be used for training
* NUMBER_OF_INFERENCE_STATIONS - number of stations (first NUMBER_OF_INFERENCE_STATIONS from ALL_STATIONS) which will be used for predicting.

### Inference
Make sure you set `DATASETS_PATH='/data'` in `amurlevel_model/config.py`
Then you must build docker image and run it with parameters
```
docker build . -f Dockerfile_predict -t raev_aij2020_amur_predict
docker run -v $(pwd)/data:/data raev_aij2020_amur_predict -f_day 2017-12-10 -w /data/weights-aij2020amurlevel-2017.h5
```

### Обучение
Make sure you set `DATASETS_PATH='/data'` in `amurlevel_model/config.py`
Then you must build docker image and run it with parameters. Model weights in this repository obtained on Kaggle GPU. The training process locally on CPU is going much slower (Intel Core I5-8265U, 4 cores, 8 GB RAM).
```
docker build . -f Dockerfile_train -t raev_aij2020_amur_train
docker run -v $(pwd)/data:/data raev_aij2020_amur_train -t_day 2017-12-01 -w /data/weights-aij2020amurlevel-2017.h5
```

### Jupyter notebook
EDA.ipynb consists of some research + code for model training and some plots. Make sure that you downloaded all necessary data and put it in `data/` directory before launch it cell by cell. If you want to train model you should set ```TRAIN=True``` in first cell.

If you want to work in jupyter notebook with the same versions of the libraries as in docker containers, you should install from requirements.txt:
```
!pip install requirements.txt
```

### Reference list
A fairly large amount of literature was proposed in preparation, below are links to the most useful (according to the author) sources (links are presented in a free format):
<ol>
<li> Спектральный анализ временных рядов в экономике [Текст] / К. Гренджер, М. Хатанака ; Пер. В. С. Дуженко, Е. Г. Угер ; Науч. ред. В. В. Налимов. - Москва : Статистика, 1972. - 312 с. : черт.; 22 см.</li>
<li> С.В. Борщ, Ю.А. Симонов, А.В. Христофоров, Н.М. Юмина. КРАТКОСРОЧНОЕ ПРОГНОЗИРОВАНИЕ УРОВНЕЙ ВОДЫ НА РЕКЕ АМУР</li>
<li> Известия Томского политехнического университета. Инжиниринг георесурсов. 2016. Т. 327. № 11. 105–115  Лариошкин В.В. Методика прогноза дождевых паводков в бассейне Верхнего Амура (на примере р. Онон)</li>
<li> Экстремальные паводки в бассейне Амура: гидрологические аспекты / Сб. работ по гидрологии / под ред. Георгиевского В.Ю., ФГБУ «ГГИ», СПБ, ООО «ЭсПэХа», 2015.- стр.171. </li>
<li> Мы и амурские наводнения: невыученный урок? / Под ред.А. В. Шаликовского. — М.: Всемирный фонд дикой природы (WWF),2016. — 216 с.</li>
<li> Калугин А.С., Модель формирования стока реки Амур и ее применение для оценки возможных изменений водного режима, дис. … канд. геогр. Наук. Институт водных проблем РАН, Москва, 2016</li>
<li> С.В. Борщ , Д.А. Бураков , Ю.А. Симонов. МЕТОДИКА ОПЕРАТИВНОГО РАСЧЕТА И ПРОГНОЗА СУТОЧНОГО ПРИТОКА ВОДЫ В ВОДОХРАНИЛИЩЕ ЗЕЙСКОЙ ГЭС </li>
<li> ЭКСТРЕМАЛЬНОЕ НАВОДНЕНИЕ В БАССЕЙНЕ АМУРА В 2013 ГОДУ: АНАЛИЗ ФОРМИРОВАНИЯ, ОЦЕНКИ И РЕКОМЕНДАЦИИ,  Болгов М.В., Алексеевский Н.И., Гарцман Б.И., Георгиевский В.Ю., Дугина И.О., Ким В.И., Махинов А.Н., Шалыгин А.Л. География и природные ресурсы. 2015. № 3. С. 17-26.</li>
 </ol>
