import mlflow
import os

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_registry_uri("http://0.0.0.0:5000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

REGISTRY_MODEL_NAME = "churn_model_barkov_v_v"

client = mlflow.MlflowClient()

models = client.search_model_versions(filter_string=f"name = '{REGISTRY_MODEL_NAME}'")
print(f"Model info:\n {models}")

model_name_1 = models[-1].name
model_version_1 = models[-1].version
model_stage_1 = models[-1].current_stage

model_name_2 = models[-2].name
model_version_2 = models[-2].version
model_stage_2 = models[-2].current_stage

print(f"Текущий stage модели 1: {model_stage_1}")
print(f"Текущий stage модели 2: {model_stage_2}")

client.transition_model_version_stage(model_name_1, model_version_1, "production")
client.transition_model_version_stage(model_name_2, model_version_2, "staging")

client.rename_registered_model(name=REGISTRY_MODEL_NAME, new_name=f"{REGISTRY_MODEL_NAME}_b2c")