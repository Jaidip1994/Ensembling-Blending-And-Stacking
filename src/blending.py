import pandas as pd
import glob
from sklearn.metrics import roc_auc_score
import numpy as np


if __name__ == '__main__':
    files = glob.glob('../model_pred/*.csv')
    df = None
    for f in files:
       if df is None:
           df = pd.read_csv(f)
       else:
           temp_df = pd.read_csv(f)
           df = df.merge(temp_df, on="id", how="left")
    targets = df.sentiment.values
    pred_col = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]

    for cols in pred_col:
        auc = roc_auc_score(targets, df[cols].values)
        print(f'{cols}, overall AUC: {auc}')
    
    # Blending -> Averaging
    print('Average')
    avg_pred = np.mean(df[pred_col].values, axis = 1)
    print(roc_auc_score(targets, avg_pred))

    # Blending -> Weighted Average
    print('Weighted Average')
    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    rf_svd_pred = df.rf_svd_pred.values
    avg_pred = ( lr_pred + 3 * lr_cnt_pred + rf_svd_pred ) / 5
    print(roc_auc_score(targets, avg_pred))

    print('Rank average')
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = ( lr_pred + lr_cnt_pred + rf_svd_pred ) / 3
    print(roc_auc_score(targets, avg_pred))

    print('Weighted Rank average')
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = ( lr_pred + 3 * lr_cnt_pred + rf_svd_pred ) / 3
    print(roc_auc_score(targets, avg_pred))
