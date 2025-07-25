
# 🫀 Heart Attack Risk Prediction Project

## 📋 Описание проекта
Проект по предсказанию риска сердечного приступа на основе медицинских показателей пациента. Модель использует различные характеристики, такие как возраст, привычки, состояние здоровья и результаты анализов крови для оценки риска сердечного приступа.

## 📁 Структура проекта

heart_project/ 
    ├── data/ # Датасеты │ 
        ├── heart_train.csv # Тренировочный набор данных │ 
        └── heart_test.csv # Тестовый набор данных 
    ├── models/ # Сохраненные модели │ 
        └── CatBoost_pipeline.joblib 
    ├── app.py # FastAPI приложение 
    ├── project.ipynb # Jupyter notebook с анализом └── requirements.txt # Зависимости проекта


## ⚡️ Основные функции
- 🔍 Загрузка и предобработка данных
- 📊 Исследовательский анализ данных (EDA)
- 🤖 Обучение различных моделей машинного обучения
- 🌐 REST API для получения предсказаний

## 🔬 Использованные модели
- 🌳 Random Forest
- 📈 Gradient Boosting
- 🎯 SVM
- 🔍 KNN
- 📊 Logistic Regression
- 🏆 CatBoost (финальная модель)

## 🌐 API Endpoints
- `GET /health` - Проверка состояния сервиса
- `POST /predict` - Предсказание для набора данных (CSV файл)
- `POST /predict_single` - Предсказание для одного пациента

## 🚀 Установка и запуск

### 1. Клонирование репозитория

bash git clone <repository-url> cd heart_project


### 2. Создание виртуального окружения

bash
# Создание
python -m venv venv
# Активация для Linux/MacOS
source venv/bin/activate
# Активация для Windows
venv\Scripts\activate


### 3. Установка зависимостей
bash pip install -r requirements.txt


### 4. Запуск Jupyter Notebook
bash jupyter notebook project.ipynb


### 5. Запуск API сервера
bash uvicorn app:app --reload


## 📡 Использование API

### Проверка состояния сервиса
bash curl [http://localhost:8000/health](http://localhost:8000/health)


### Предсказание для одного пациента
bash curl -X POST "[http://localhost:8000/predict_single](http://localhost:8000/predict_single)"
-H "Content-Type: application/json"
-d '{ "features": { "Age": 45, "Gender": 1, "Cholesterol": 230, "Heart rate": 80 } }'


### Пакетное предсказание
bash curl -X POST "[http://localhost:8000/predict](http://localhost:8000/predict)"
-F "file=@path/to/your/data.csv"


## 📚 Документация API
После запуска сервера документация доступна по адресам:
- 📘 Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📗 ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📦 Зависимости
- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- CatBoost
- Joblib
- NumPy
- Matplotlib
- Seaborn
- Jupyter

## 📊 Анализ данных
Полный анализ данных в `project.ipynb` включает:
- ✨ Обработку пропущенных значений
- 📈 Корреляционный анализ
- 📊 Визуализацию распределений
- 🎯 Анализ важности признаков

## 📈 Метрики качества
- ROC AUC
- Accuracy
- F1-score

_Конкретные значения доступны в ноутбуке_

## 📄 Лицензия
MIT

## 👨‍💻 Авторы
[Артем Казаков]

