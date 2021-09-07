import glob
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from functools import partial
from scipy.optimize import fmin

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def run_training(pred_df, fold, pred_cols):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)
    xtrain = train_df[pred_cols]
    xvalid = valid_df[pred_cols]

    # scl = StandardScaler()
    # xtrain = scl.fit_transform(xtrain)
    # xvalid = scl.transform(xvalid)
    opt = LinearRegression()
    opt.fit(xtrain, train_df.target.values)
    pred = opt.predict(xvalid)
    rmse = mean_squared_error(valid_df.target.values, pred, squared=False)
    valid_df.loc[:, 'opt_pred'] = pred
    print(f'Fold : {fold}, RMSE : {rmse}')
    return opt.coef_


files = glob.glob('../data/*cnt*.csv')
df = None
for f in files:
    if df is None:
        df = pd.read_csv(f)
        df.sort_index(inplace = True)
    else:
        temp_df = pd.read_csv(f)
        temp_df.sort_index(inplace = True)
        # df = df.merge(temp_df[['index', temp_df.columns.tolist()[-1]]], on="index", how="left")
        df.loc[:, temp_df.columns.tolist()[-1]] = temp_df[temp_df.columns.tolist()[-1]] 
df.to_csv('test.csv')
df.head()
targets = df.target.values
pred_col = ['cat_cnt_pred', 'lasso_cnt_pred',
       'lgbm_cnt_pred', 'lr_cnt_pred', 'rf_cnt_pred', 'ridge_cnt_pred',
       'xgb_cnt_pred']

for cols in pred_col:
    rmse = mean_squared_error(targets, df[cols].values, squared=False)
    print(f'{cols}, overall rmse: {rmse}')

coefs = []
for j in range(5):
    coefs.append(run_training(df, j, pred_col))
coefs = np.array(coefs)
coefs = np.mean(coefs, axis = 0)

print(coefs)
wt_avg = coefs[0] * df.cat_cnt_pred.values + coefs[1] * df.lasso_cnt_pred.values + coefs[2] * df.lgbm_cnt_pred.values 
+ coefs[3] * df.lr_cnt_pred.values + coefs[4] * df.rf_cnt_pred.values + coefs[5] * df.ridge_cnt_pred.values + coefs[6] * df.xgb_cnt_pred.values 
print(wt_avg[:5])
print('Optimal rmse')
print(mean_squared_error(targets, wt_avg, squared=False))
