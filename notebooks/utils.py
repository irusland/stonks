import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def stock_option_accuracy_score(df, y_true, y_pred):
    true_pred_by_date = pd.concat([df.iloc[y_true.index].date, y_true, y_pred], axis=1).groupby('date').mean()
    true_pred_by_date_prev = true_pred_by_date.iloc[:-1]
    true_pred_by_date_curr = true_pred_by_date.iloc[1:]
    true_pred_diff = true_pred_by_date_curr.to_numpy() - true_pred_by_date_prev.to_numpy()
    true_pred_binary = true_pred_diff > 0
    true_binary = true_pred_binary[:,0]
    pred_binary = true_pred_binary[:,1]
    return accuracy_score(true_binary, pred_binary)


def stock_possible_profit_percent_score(df, y_true, y_pred):
    true_pred_by_date = pd.concat([df.iloc[y_true.index].date, y_true, y_pred], axis=1).groupby('date').mean()
    true_pred_by_date_prev = true_pred_by_date.iloc[:-1]
    true_pred_by_date_curr = true_pred_by_date.iloc[1:]
    true_pred_diff = true_pred_by_date_curr.to_numpy() - true_pred_by_date_prev.to_numpy()
    true_pred_binary = true_pred_diff > 0
    true_binary = true_pred_binary[:,0]
    pred_binary = true_pred_binary[:,1]
    profit_percent = np.abs(1 - (true_pred_by_date_curr.to_numpy() / true_pred_by_date_prev.to_numpy())[:,0])
    percent_changes = profit_percent * np.where((true_binary==pred_binary), 1, -1)

    return 100 * (np.prod(1 + percent_changes) - 1)
