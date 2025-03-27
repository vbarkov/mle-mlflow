import mlflow
import os
from mlflow_experiment import start_experiment_run
from data_loader import load_data

OUT_DIR = "../out"
COLUMNS_INFO_PATH = f"{OUT_DIR}/columns.txt"
DATASET_PATH = f"{OUT_DIR}/users_churn.csv"

os.makedirs(OUT_DIR, exist_ok=True)

df = load_data()

with open(COLUMNS_INFO_PATH, "w", encoding="utf-8") as fio:
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

df.to_csv(DATASET_PATH, index=False) 

EXPERIMENT_NAME = "churn_barkov_v_v"
RUN_NAME = "data_check"

def run_entry_point(run):   
    # логируем метрики эксперимента
    # предполагается, что переменная stats содержит словарь с метриками,
    # где ключи — это названия метрик, а значения — числовые значения метрик
    mlflow.log_metrics(stats)
    
    # логируем файлы как артефакты эксперимента — 'columns.txt' и 'users_churn.csv'
    mlflow.log_artifact(COLUMNS_INFO_PATH, "dataframe")
    mlflow.log_artifact(DATASET_PATH, "dataframe")

    #os.remove(COLUMNS_INFO_PATH)
    #os.remove(DATASET_PATH)

run_id = start_experiment_run(EXPERIMENT_NAME, RUN_NAME, run_entry_point)  

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
# получаем данные о запуске эксперимента по его уникальному идентификатору
run = mlflow.get_run(run_id)

# проверяем, что статус запуска эксперимента изменён на 'FINISHED'
# это утверждение (assert) можно использовать для автоматической проверки того, 
# что эксперимент был завершён успешно
assert run.info.status=="FINISHED"