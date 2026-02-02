import pandas as pd
import numpy as np

def  remake_monthincome(row: pd.Series) -> float:
    """
    Выполняет импьютацию пропущенных значений в столбце 'MonthlyIncome' на основе среднего
    дохода заёмщиков в возрастном диапазоне ±3 года от текущего возраста строки.
    Если значение 'MonthlyIncome' не пропущено, возвращается исходное значение.

    Аргументы:
        row (pd.Series): Одна строка DataFrame, содержащая все признаки, включая
                        'MonthlyIncome' и 'age'.

    Возвращаемое значение:
        float: Среднее значение дохода для возрастного диапазона, если 'MonthlyIncome'
               является NaN, иначе исходное значение 'MonthlyIncome'.
    """
    if pd.isnull(row['MonthlyIncome']):
        avg_income = df[(df['age'] >= row['age'] - 3) & (df['age'] <= row['age'] + 3)]['MonthlyIncome'].mean()
        return avg_income
    else:
        return row['MonthlyIncome']



def remake_monthly_debt_payments(row: pd.Series) -> float:
    """
    Выполняет импьютацию пропущенных значений в столбце 'MonthlyDebtPayments' на основе
    среднего значения ежемесячных платежей заёмщиков в возрастном диапазоне ±3 года
    от текущего возраста строки. Если значение 'MonthlyDebtPayments' не пропущено,
    возвращается исходное значение.

    Аргументы:
        row (pd.Series): Одна строка DataFrame, содержащая все признаки, включая
                        'MonthlyDebtPayments' и 'age'.

    Возвращаемое значение:
        float: Среднее значение ежемесячных платежей для возрастного диапазона, если
               'MonthlyDebtPayments' является NaN, иначе исходное значение 'MonthlyDebtPayments'.
    """
    if pd.isnull(row['MonthlyDebtPayments']):
        avg_debt = df[(df['age'] >= row['age'] - 3) & (df['age'] <= row['age'] + 3)]['MonthlyDebtPayments'].mean()
        return avg_debt
    else:
        return row['MonthlyDebtPayments']

def balance(df: pd.DataFrame)-> pd.DataFrame:
    """
    Выполняет балансировку набора данных по целевой переменной 'SeriousDlqin2yrs',
    уравнивая количество примеров положительного класса (1) и отрицательного класса (0).
    Отрицательный класс (0) случайным образом подбирается до количества, равного числу
    примеров положительного класса (1), с сохранением порядка.

    Аргументы:
        df (pd.DataFrame): Исходный DataFrame, содержащий столбец 'SeriousDlqin2yrs'
                          с целевой переменной (0 или 1).

    Возвращаемое значение:
        pd.DataFrame: Сбалансированный DataFrame, где количество строк с 'SeriousDlqin2yrs == 1'
                      равно количеству строк с 'SeriousDlqin2yrs == 0', с перемешанными
                      строками и сброшенными индексами.
    """
    num = df['SeriousDlqin2yrs'].value_counts()[1]
    df_t = df[df.SeriousDlqin2yrs==1]
    df_f = df[df.SeriousDlqin2yrs==0].sample(frac=1,random_state=21)[0:num]
    df = pd.concat([df_t,df_f]).sample(frac=1).reset_index(drop=True)

    return df

def preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    df = df.drop('Unnamed: 0', axis=1)
    df.drop('NumberOfTime60-89DaysPastDueNotWorse', axis = 1)
    median_age = df[df['age'] < 90]['age'].median()
    df.loc[df['age'] >= 90, 'age'] = median_age
    df = df[df['NumberOfTime30-59DaysPastDueNotWorse'] < 96]
    df['MonthlyIncome'] = df.apply(remake_monthincome, axis=1)
    df['MonthlyDebtPayments'] = np.where(
        (df['DebtRatio'] <= 1) & (df['MonthlyIncome'] > 1),
        df['DebtRatio'] * df['MonthlyIncome'],
        np.nan
    )
    df['MonthlyDebtPayments'] = df.apply(remake_monthly_debt_payments, axis=1)
    df['MonthlyDebtPayments'] = df['MonthlyDebtPayments'].fillna(df['MonthlyDebtPayments'].median())
    df = df[df['DebtRatio'] <= 1]
    df['NumberOfDependents'] = df['NumberOfDependents'].replace({10: 1, 20: 2})
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0]).astype(int)
    
    Q1 = df['MonthlyIncome'].quantile(0.25)
    Q3 = df['MonthlyIncome'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    high_bound = Q3 + 3 * IQR
    print(f'Нижняя граница: {lower_bound}')
    print(f"Вверхняя граница: {high_bound}")
    df = df[df['MonthlyIncome'] <= 30000]
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] < 20000]
    df = df[df['NumberRealEstateLoansOrLines']<20]
    df = balance(df)
    return df