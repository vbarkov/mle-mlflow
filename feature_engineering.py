import os
from catboost import CatBoostClassifier

import pandas as pd
import psycopg
import mlflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

TABLE_NAME = "clean_users_churn"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_barkov_v_v"
RUN_NAME = "preprocessing" 
REGISTRY_MODEL_NAME = "churn_model_barkov_v_v"

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"), 
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
connection.update(postgres_credentials)
# эта конструкция создаёт контекстное управление для соединения с базой данных 
# оператор with гарантирует, что соединение будет корректно закрыто после выполнения всех операций 
# закрыто оно будет даже в случае ошибки, чтобы не допустить "утечку памяти"
with psycopg.connect(**connection) as conn:

# создаёт объект курсора для выполнения запросов к базе данных
# с помощью метода execute() выполняется SQL-запрос для выборки данных из таблицы TABLE_NAME
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
                
                # извлекаем все строки, полученные в результате выполнения запроса
        data = cur.fetchall()

                # получает список имён столбцов из объекта курсора
        columns = [col[0] for col in cur.description]

# создаёт объект DataFrame из полученных данных и имён столбцов. 
# это позволяет удобно работать с данными в Python, используя библиотеку Pandas.
df = pd.DataFrame(data, columns=columns)

obj_df = df.select_dtypes(include="object")

print(obj_df.info())

cat_columns = ["type", "payment_method", "internet_service", "gender"]


# создание объекта OneHotEncoder для преобразования категориальных переменных
# auto - автоматическое определение категорий
# ignore - игнорировать ошибки, если встречается неизвестная категория
# max_categories - максимальное количество уникальных категорий
# sparse_output - вывод в виде разреженной матрицы, если False, то в виде обычного массива
# drop="first" - удаляет первую категорию, чтобы избежать ловушки мультиколлинеарности
encoder_oh = OneHotEncoder(categories="auto",
                           handle_unknown="ignore",
                           max_categories=10,
                           drop="first",
                           sparse_output=False
)

# применение OneHotEncoder к данным. Преобразование категориальных данных в массив
encoded_features = encoder_oh.fit_transform(df[cat_columns].to_numpy())

# преобразование полученных признаков в DataFrame и установка названий колонок
# get_feature_names_out() - получение имён признаков после преобразования
encoded_df = pd.DataFrame(data=encoded_features, columns=encoder_oh.get_feature_names_out())

# конкатенация исходного DataFrame с новым DataFrame, содержащим закодированные категориальные признаки
# axis=1 означает конкатенацию по колонкам
obj_df = pd.concat([obj_df, encoded_df], axis=1)

obj_df.head(2)


num_columns = ["monthly_charges", "total_charges"]
num_df = df[num_columns]
n_knots = 3
degree_spline = 4
n_quantiles=100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
subsample = None


# SplineTransformer
encoder_spl = SplineTransformer(n_knots=n_knots,
                               degree=degree_spline
)
encoded_features = encoder_spl.fit_transform(df[num_columns].to_numpy())
encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_spl.get_feature_names_out(num_columns)
)
num_df = pd.concat([num_df, encoded_df], axis=1)


# QuantileTransformer
encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
encoded_features = encoder_q.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_q.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_q_{n_quantiles}" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)


# RobustScaler
encoder_rb = RobustScaler()
encoded_features = encoder_rb.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_rb.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_robust" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)


# PolynomialFeatures
encoder_pol = PolynomialFeatures(degree=degree)
encoded_features = encoder_pol.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_pol.get_feature_names_out(num_columns)
)
encoded_df.drop(encoded_df.columns[:1 + len(num_columns)], axis=1, inplace=True)
num_df = pd.concat([num_df, encoded_df], axis=1)

# KBinsDiscretizer
encoder_kbd = KBinsDiscretizer(n_bins=n_bins,
                               encode=encode,
                               strategy=strategy,
                               subsample=subsample)
encoded_features = encoder_kbd.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder_kbd.get_feature_names_out(num_columns)
)
encoded_df.columns = [col + f"_bin" for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

num_df.head(2)


numeric_transformer = ColumnTransformer(
    transformers=[
        ('spl', encoder_spl, num_columns), 
        ('q', encoder_q, num_columns), 
        ('rb', encoder_rb, num_columns), 
        ('pol', encoder_pol, num_columns), 
        ('kbd', encoder_kbd, num_columns)
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", encoder_oh)
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_columns), 
        ('cat', categorical_transformer, cat_columns)], 
    n_jobs=-1
)

encoded_features = preprocessor.fit_transform(df)

transformed_df = pd.DataFrame(encoded_features, columns=preprocessor.get_feature_names_out())
df = pd.concat([df, transformed_df], axis=1)
df.head(2)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    print(run_id)
    mlflow.sklearn.log_model(preprocessor, "column_transformer") 

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target", "customer_id", "id"]), 
        df["target"], 
        test_size=0.2,
        random_state=42)
    metrics = {}
    model = CatBoostClassifier(auto_class_weights='Balanced')

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]

    # посчитайте метрики из модуля sklearn.metrics
    # err_1 — ошибка первого рода
    # err_2 — ошибка второго рода
    _, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()
    auc = roc_auc_score(y_test, probas)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    logloss = log_loss(y_test, prediction)

    metrics["err1"] = err1
    metrics["err2"] = err2
    metrics["auc"] = auc
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["logloss"] = logloss

    mlflow.log_metrics(metrics)

    pip_requirements = "requirements.txt"
    print("X_test:")
    print(X_test.info())
    print("X_test:")
    print(X_test)
    print("y_test:")
    print(y_test)

    signature = mlflow.models.infer_signature(X_test, y_test)
    input_example = X_test[:10]
    metadata = {"model_type": "monthly"}
    print(os.environ["AWS_ACCESS_KEY_ID"])
    print(os.environ["AWS_SECRET_ACCESS_KEY"])
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
        metadata=metadata,
        await_registration_for=60
    )

    loaded_model = mlflow.sklearn.load_model(model_uri=model_info.model_uri)
    model_predictions = loaded_model.predict(X_test)
    assert model_predictions.dtype == int

    print(model_predictions[:10])