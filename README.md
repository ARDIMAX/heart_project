``` markdown
# README.md
# Heart Attack Risk Prediction Project

## Описание проекта
Проект по предсказанию риска сердечного приступа на основе различных медицинских показателей пациента.

## Структура проекта
```
heart_project/ ├── data/ │ ├── heart_train.csv │ └── heart_test.csv ├── models/ │ └── saved_models/ ├── app.py ├── models.py ├── test_api.py └── requirements.txt
``` 

## Установка и запуск

1. Установка зависимостей:
```
bash pip install -r requirements.txt
``` 

2. Обучение моделей:
```
bash python models.py
``` 

3. Запуск API:
```
bash uvicorn app:app --reload
``` 

4. Запуск тестов:
```
bash pytest test_api.py
``` 

## API Endpoints

- POST `/predict` - Предсказание для набора данных (CSV файл)
- POST `/predict_single` - Предсказание для одного пациента

## Использованные модели
- Random Forest
- Gradient Boosting
- SVM
- KNN
- Logistic Regression
- CatBoost

## Метрики качества
[Здесь будут метрики после обучения моделей]
```

``` txt
# requirements.txt
fastapi==0.68.1
uvicorn==0.15.0
pandas==1.3.3
scikit-learn==0.24.2
catboost==0.26.1
joblib==1.0.1
pytest==6.2.5
httpx==0.19.0
python-multipart==0.0.5
```
Этот код:
1. Обучает и сравнивает 6 разных моделей
2. Сохраняет лучшую модель
3. Предоставляет API для предсказаний
4. Включает тесты
5. Содержит полную документацию

Для использования:
1. Создайте структуру проекта
2. Установите зависимости: `pip install -r requirements.txt`
3. Запустите обучение моделей
4. Запустите API: `uvicorn app:app --reload`
5. Используйте эндпоинты для предсказаний

API поддерживает как пакетные предсказания (через CSV файл), так и предсказания для отдельных пациентов.
