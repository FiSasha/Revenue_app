import streamlit as st
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import tensorflow as tf

from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error

xgb_model_path = "./models/xgb_model.pkl"
lstm_model_path = "./models/lstm_model.h5"
dataset_path = "./data/dateTime.csv"

def main():
    page = st.sidebar.selectbox("Страница", ("Инфо", "Модели"), key = "Select")

    if (page == "Инфо"):
        st_about()
    elif (page == "Модели"):
        st_prediction_page()
    else:
        pass

def st_prediction_page():
    model_name = st.radio("Выберите модель для тестирования", ["XGB", "LSTM"])
    dataset_df, feature, target = func_prepare_data(dataset_path)
    xgb_rmse, xgb_rmsle, xgb_x, xgb_y_actual, xgb_y_predicted = func_make_predictions("XGB", dataset_df, feature, target)
    lstm_rmse, lstm_rmsle, lstm_x, lstm_y_actual, lstm_y_predicted = func_make_predictions("LSTM", dataset_df, feature, target)
    if model_name == "XGB":
        func_prepare_text_and_plot(model_name, xgb_rmse, xgb_rmsle, xgb_x, xgb_y_actual, xgb_y_predicted)
    elif model_name == "LSTM":
        func_prepare_text_and_plot(model_name, lstm_rmse, lstm_rmsle, lstm_x, lstm_y_actual, lstm_y_predicted)
    else:
        pass
    

def st_about():
    st.header("О работе")
    st.markdown("""
        Добро пожаловать в приложение по прогнозу будущей выручки.

        Это приложение предназначено для анализа данных о продажах. Вы можете выбрать одну из двух предложенных моделей для прогноза и на основании метрики качества (LRSME) выбрать наиболее подходящую.
        
        Для того, чтобы перейти к моделям, выберите на панели слева из выпадающего списка страницу **"Модели"**.
    """)

def func_make_features(data, max_lag, rolling_mean_size):
    data['month'] = data.index.month
    data['dayofweek'] = data.index.dayofweek
    data['year'] = data.index.year

    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['Revenue'].shift(lag)

    data['rolling_mean'] = data['Revenue'].shift().rolling(rolling_mean_size).mean()
    return data

def func_create_sequences(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def func_prepare_text_and_plot(model_name, rmse, rmsle, x, y_actual, y_predicted):
    st.markdown(f"""
            RMSE {model_name} на тесте: {rmse:.5f}  
            RMSLE {model_name} на тесте: {rmsle:.5f}  
    """)
    fig = plt.figure(figsize=(12, 6))
    plt.title(f'Revenue Forecasting with {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.plot(x, y_actual, label='Actual Revenue', color='blue')
    plt.plot(x, y_predicted, label='Predicted Revenue', color='red')
    plt.legend()
    st.pyplot(fig)

def func_prepare_data(dataset_path):
    dataset_df = pd.read_csv(dataset_path)
    dataset_df['Date'] = pd.to_datetime(dataset_df['Date'])
    dataset_df = dataset_df.set_index('Date')
    dataset_df_dl = dataset_df[['Revenue']].copy()
    dataset_df_prep = dataset_df[['Revenue']].copy()
    dataset_df_ml = func_make_features(dataset_df_prep, 6, 3)
    dataset_df_ml = dataset_df_ml.dropna()
    feature = dataset_df_ml.drop('Revenue', axis=1)
    target = dataset_df_ml['Revenue']
    return dataset_df_dl, feature, target

def func_make_predictions(model_name, dataset_df, feature, target):
    if model_name == "XGB":
        model = joblib.load(xgb_model_path)

        _, X_test, _, y_test = train_test_split(feature, target, test_size=0.2, random_state=142, shuffle=False)
        predictions = model.predict(X_test)

        rmse = mean_squared_error(predictions, y_test, squared=False)
        rmsle = mean_squared_log_error(predictions, y_test, squared=False)

        x = feature.index
        y_actual = target
        y_predicted = model.predict(feature)
    elif model_name == "LSTM":
        model = tf.keras.models.load_model(lstm_model_path)

        train_dl, test_dl = train_test_split(dataset_df, test_size=0.2, random_state=142, shuffle=False)
        scaler = MinMaxScaler()
        scaled_train_data = scaler.fit_transform(train_dl)
        scaled_test_data = scaler.transform(test_dl)
        look_back = 6
        trainX, _ = func_create_sequences(scaled_train_data, look_back)
        testX, testY = func_create_sequences(scaled_test_data, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        testPredict = model.predict(testX)
        testPredictLSTM = scaler.inverse_transform(testPredict)
        test_dates = test_dl.index[look_back:]
        testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

        rmse = mean_squared_error(testY_inverse, testPredictLSTM, squared=False)
        rmsle = mean_squared_log_error(testY_inverse, testPredictLSTM, squared=False)

        x = test_dates
        y_actual = testY_inverse.flatten()
        y_predicted = testPredictLSTM.flatten()
    return rmse, rmsle, x, y_actual, y_predicted

main()
