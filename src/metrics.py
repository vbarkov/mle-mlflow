from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    log_loss, 
    confusion_matrix
)


def calculate_metrics(y_test, prediction, probas):
    _, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()
    return {
        "err1": err1,
        "err2": err2,
        "auc": roc_auc_score(y_test, probas),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "logloss": log_loss(y_test, prediction) 
    }
