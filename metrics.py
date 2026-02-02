from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report

def show_metrics(y_true, y_pred, y_pred_proba):
    print('Test Metrics:')
    print(f'AUC-ROC: {roc_auc_score(y_true, y_pred_proba):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall: {recall_score(y_true, y_pred):.4f}')
    print(f'F1: {f1_score(y_true, y_pred):.4f}')
    print('\nClassification Report:\n', classification_report(y_true, y_pred))