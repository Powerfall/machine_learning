import os

import pandas as pd
import streamlit as st
from keras.models import load_model


def show_title_with_subtitle():
    # Заголовок и подзаголовок
    st.title("Распознование рукописных цифр")


def show_info_page():
    st.write("## Задача")
    st.write(
        "MNIST («Модифицированный национальный институт стандартов и технологий») — это фактически набор данных компьютерного зрения уровня «hello world». "
        "С момента своего выпуска в 1999 году этот классический набор данных рукописных изображений служил основой для алгоритмов классификации бенчмаркинга. "
        "По мере появления новых методов машинного обучения MNIST остается надежным ресурсом как для исследователей, так и для учащихся.")
    st.image("https://docs.microsoft.com/ru-ru/azure/databricks/_static/images/distributed-deep-learning/mnist.png",
             use_column_width=True)
    st.write("Цель — правильно идентифицировать цифры из набора данных.")
    st.write("### Выбранная модель")
    st.write(
        "В результате анализа метрик качества нескольких классификационных моделей выбрана модель"
        "Keras (https://keras.io/about/), обеспечивающая"
        "высокое качество предсказаний рукописных цифр.")
    st.write("Выполненная работа представляет собой результат участия в соревновании на портале Kaggle. Более подробно"
             "ознакомиться с правилами соревнования можно по ссылке ниже:")
    st.write("https://www.kaggle.com/c/digit-recognizer/overview/description")


def show_predictions_page():
    st.write("Файл для примера: https://drive.google.com/file/d/1p_TKAl8XpdFekRFx7FGl8ACYtMkzVvRG/view?usp=sharing")
    file = st.file_uploader(label="Выберите csv файл с предобработанными данными для прогнозирования цифр",
                            type=["csv"],
                            accept_multiple_files=False)
    if file is not None:
        test_data = pd.read_csv(file)
        st.write("### Загруженные данные")
        st.write(test_data)
        test_data = test_data / 255.0
        test_data = test_data.values.reshape(-1, 28, 28, 1)
        make_predictions(get_model(), test_data)


def get_model():
    return load_model(os.path.join(os.path.dirname(__file__), "models", "c_model_1"))


def make_predictions(model, X):
    st.write("### Предсказанные значения")
    pred = pd.DataFrame(model.predict(X))
    st.write(pred)


def select_page():
    # Сайдбар для смены страницы
    return st.sidebar.selectbox("Выберите страницу", ("Информация", "Прогнозирование"))


# Стиль для скрытия со страницы меню и футера streamlit
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# размещение элементов на странице
show_title_with_subtitle()
st.sidebar.title("Меню")
page = select_page()
st.sidebar.write("© Lorents Nikita 2021")
st.sidebar.write("https://github.com/Powerfall")

if page == "Информация":
    show_info_page()
else:
    show_predictions_page()
