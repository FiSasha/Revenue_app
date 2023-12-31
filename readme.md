# О приложении

Приложение написано на Python с использованием библиотеки для веб-приложений streamlit. В файле requirements.txt вы можете найти список библиотек, необходимых для работы приложения. Для приложения подготовлен Dockerfile с описанием образа, который можно собрать и запустить приложение в контейнере.
Структура репозитория:
- data: каталог с датасетом (файл dateTime.csv) 
- models: каталог с сохраненными файлами обученных моделей (pkl формат для XGBoost модели и h5 формат для LSTM модели)
- app.py: основной Python скрипт с кодом streamlit приложения и кодом для запуска теста каждой из моделей, а также отрисовки графиков
- Dockerfile: файл с описанием Docker образа с приложением
- requirements.txt: файл с перечислением необходимых Python библиотек
- readme.md: файл с кратким описанием проекта

Для сборки образа запустите следующую команду:
```
docker build -t revenue_app .
```
Для запуска контейнера запустите команду: 
```
docker run --rm -p 8501:8501 revenue_app
```
Приложение будет доступно по адресу: 
```
localhost:8501
```