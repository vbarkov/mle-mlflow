import mlflow
import os
import pandas as pd
import psycopg

mlflow.set_tracking_uri("http://0.0.0.0:5000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"), 
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
assert all([var_value != "" for var_value in list(postgres_credentials.values())])
print(postgres_credentials)
connection.update(postgres_credentials)

# определим название таблицы, в которой хранятся наши данные.
TABLE_NAME = "users_churn"

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

with open("columns.txt", "w", encoding="utf-8") as fio:
    fio.write(",".join(df.columns))

counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
    # посчитайте уникальные значения для колонок, где немного уникальных значений (переменная counts_columns)
    column_stat = df[col].value_counts()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}

    # обновите словарь stats
    stats.update(column_stat)

stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max()
stats["monthly_charges_mean"] = df["monthly_charges"].mean()
stats["monthly_charges_median"] = df["monthly_charges"].median()
stats["total_charges_min"] = df["total_charges"].min()
stats["total_charges_max"] = df["total_charges"].max()
stats["total_charges_mean"] = df["total_charges"].mean()
stats["total_charges_median"] = df["total_charges"].median()
stats["unique_customers_number"] = len(df["customer_id"].unique())
stats["end_date_nan"] = df["end_date"].isnull().sum()

df.to_csv("users_churn.csv", index=False) 

EXPERIMENT_NAME = "churn_barkov_v_v"
RUN_NAME = "data_check"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is not None:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    # получаем уникальный идентификатор запуска эксперимента
    run_id = run.info.run_id
    
    # логируем метрики эксперимента
    # предполагается, что переменная stats содержит словарь с метриками,
    # где ключи — это названия метрик, а значения — числовые значения метрик
    mlflow.log_metrics(stats)
    
    # логируем файлы как артефакты эксперимента — 'columns.txt' и 'users_churn.csv'
    mlflow.log_artifact("columns.txt", "dataframe")
    mlflow.log_artifact("users_churn.csv", "dataframe")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
# получаем данные о запуске эксперимента по его уникальному идентификатору
run = mlflow.get_run(run_id)


# проверяем, что статус запуска эксперимента изменён на 'FINISHED'
# это утверждение (assert) можно использовать для автоматической проверки того, 
# что эксперимент был завершён успешно
assert run.info.status=="FINISHED"

os.remove("columns.txt")
os.remove("users_churn.csv")