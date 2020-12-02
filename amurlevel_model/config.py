# -*- coding: utf-8 -*-

# путь до файла с датасетом
DATASETS_PATH = '/data'

# количество прогнозируемых дней вперед. Менять только с переобучением модели!
DAYS_FORECAST = 10

# все станции, информация с которых будет использоваться в модели. Менять только с переобучением модели!
ALL_STATIONS = ('05004', '05012', '05013', '05024', '05805', '06005', '06022', '06027',
                '06296', '05001', '05002', '05009', '05016', '05019', '05020', '05026',
                '05029', '05031', '05033', '05044', '05336', '05352', '05354', '05454',
                '05803', '06010', '06020', '06024', '06026', '06030', '06256', '06259',
                '06549')

# количество первых по списку станций из ALL_STATIONS, для которых выводим результат в инференсе
NUMBER_OF_INFERENCE_STATIONS = 8

MAX_DATE = '2100-01-01' # максимальная дата, после которой данные считаем невалидными

# яндекс.погода API
YANDEX_KEY = '' # ключ для доступа к сервису
YANDEX_URL = 'https://api.weather.yandex.ru/v2/forecast' # урл для API

# open-weather API
OPEN_WEATHER_KEY = '' # ключ для доступа к сервису
OPEN_WEATHER_URL = 'https://community-open-weather-map.p.rapidapi.com/forecast/daily' # урл для API
OPEN_WEATHER_HOST = 'community-open-weather-map.p.rapidapi.com' # требуется в хидерах при обращении к API

# asunp API
ASUNP_API_URL = 'http://asunp.meteo.ru/geoits-rest/services/asunp/geo.json' # урл для API

# модель
BATCH_SIZE = 8 # размер батча
EPOCHS = 100 # количество эпох

