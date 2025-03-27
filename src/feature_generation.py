from sklearn.pipeline import Pipeline
from data_loader import load_data
from sklearn.model_selection import train_test_split
from  autofeat import AutoFeatClassifier
import mlflow
from mlflow_experiment import start_experiment_run
from catboost import CatBoostClassifier
from preprocessor import create_preprocessor
from metrics import calculate_metrics

df = load_data()

cat_features = [
    'paperless_billing',
    'payment_method',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'gender',
    'senior_citizen',
    'partner',
    'dependents',
    'multiple_lines',
]
num_features = ["monthly_charges", "total_charges"]

features = cat_features + num_features

target = ["target"]

split_column = "begin_date"
test_size = 0.2

df = df.sort_values(by=[split_column])

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=test_size,
    shuffle=False,
)

transformations = ("1/", "log", "abs", "sqrt")

afc = AutoFeatClassifier(
    categorical_cols=cat_features, 
    transformations=transformations,
    feateng_steps=1,
    n_jobs=-1
)

X_train_features = afc.fit_transform(X_train, y_train)
X_test_features = afc.transform(X_test)

EXPERIMENT_NAME = "churn_barkov_v_v"
FEATURE_GENERATION_RUN_NAME = "feature_autogeneration"
REGISTRY_MODEL_NAME = "churn_model_barkov_v_v"

def feature_generation_run_entry_point(run: mlflow.ActiveRun):
    artifact_path = "afc"
    afc_info = mlflow.sklearn.log_model(afc, artifact_path=artifact_path)

start_experiment_run(EXPERIMENT_NAME, FEATURE_GENERATION_RUN_NAME, feature_generation_run_entry_point)

def fitting_model_run_entry_point(run: mlflow.ActiveRun):
    preprocessor = create_preprocessor()
    model = CatBoostClassifier(auto_class_weights='Balanced')

    pipeline = Pipeline(
        steps=[
            #("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train_features, y_train)

    prediction = pipeline.predict(X_test_features)
    probas = pipeline.predict_proba(X_test_features)[:, 1]

    metrics = calculate_metrics(y_test, prediction, probas)

    mlflow.log_metrics(metrics)

    pip_requirements = "../requirements.txt"
    signature = mlflow.models.infer_signature(X_test_features, y_test)
    input_example = X_test_features[:10]
    metadata = {"model_type": "monthly"}

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

FITTING_MODEL_RUN_NAME = "fit_model_with_generated_features"

start_experiment_run(EXPERIMENT_NAME, FITTING_MODEL_RUN_NAME, fitting_model_run_entry_point)
