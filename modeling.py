import pandas as pd
from sklearn.metrics import precision_score,roc_auc_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
import optuna


def learning_model(df: pd.DataFrame):
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8, random_state=21)

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Определяет целевую функцию для оптимизации гиперпараметров модели CatBoost с использованием
        Optuna. Функция настраивает параметры модели, обучает её на тренировочных данных и
        возвращает значение метрики precision_score на валидационной выборке.

        Аргументы:
            trial (optuna.trial.Trial): Объект Optuna для предложения значений гиперпараметров.

        Возвращаемое значение:
            float: Значение метрики precision_score на валидационной выборке для оценки качества
                модели с заданными гиперпараметрами.
        """
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1500),
            'depth': trial.suggest_int('depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 25),
            'random_seed': 42,
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
        y_pred = model.predict(X_val)
        return precision_score(y_val, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print('Best params:', best_params)

    model = CatBoostClassifier(**best_params, verbose=200)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    model.save_model('catboost_model.cbm')
    return (X_test, y_test)
