from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from data_loader import load_data
from mlflow_experiment import start_experiment_run
import mlflow
import os
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
RUN_NAME = "fit_model"
REGISTRY_MODEL_NAME = "churn_model_barkov_v_v"
model_predictions = None

def fit_model(run):
    # логируем метрики эксперимента
    # предполагается, что переменная stats содержит словарь с метриками,
    # где ключи — это названия метрик, а значения — числовые значения метрик
    mlflow.log_metrics(stats)
    for col in df.select_dtypes(include='object').columns.tolist():
        if (col != "target"):
            df[col] = df[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target", "customer_id", "id"]), 
        df["target"], 
        test_size=0.2,
        random_state=42)
    metrics = {}

    cat_features = X_train.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X_train.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
        ('binary', OneHotEncoder(drop='if_binary'), binary_cat_features.columns.tolist()),
        ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
        ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

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

    pip_requirements = "../requirements.txt"
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

    # логируем файлы как артефакты эксперимента — 'columns.txt' и 'users_churn.csv'
    mlflow.log_artifact(COLUMNS_INFO_PATH, "dataframe")
    mlflow.log_artifact(DATASET_PATH, "dataframe")

    #os.remove(COLUMNS_INFO_PATH)
    #os.remove(DATASET_PATH)

run_id = start_experiment_run(EXPERIMENT_NAME, RUN_NAME, fit_model)

# получаем данные о запуске эксперимента по его уникальному идентификатору
run = mlflow.get_run(run_id)

# проверяем, что статус запуска эксперимента изменён на 'FINISHED'
# это утверждение (assert) можно использовать для автоматической проверки того, 
# что эксперимент был завершён успешно
assert run.info.status=="FINISHED"