import pandas as pd
import glob
from sklearn.metrics import roc_auc_score
import numpy as np
from functools import partial
from scipy.optimize import fmin

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0
    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis = 1)
        auc_score = roc_auc_score(y, predictions)
        return -1.0 * auc_score
    def fit(self, X, y):
        partial_loss = partial(self._auc, X= X, y = y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp = True)
    
    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis = 1)
        return predictions

def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)
    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values

    opt = OptimizeAUC()
    opt.fit(xtrain, train_df.sentiment.values)
    pred = opt.predict(xvalid)
    auc = roc_auc_score(valid_df.sentiment.values, pred)
    valid_df.loc[:, 'opt_pred'] = pred
    print(f'Fold : {fold}, AUC : {auc}')
    return opt.coef_

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
    coefs = []
    for j in range(5):
        coefs.append(run_training(df, j))
    coefs = np.array(coefs)
    coefs = np.mean(coefs, axis = 0)
    
    wt_avg = coefs[0] * df.lr_pred.values + coefs[1] * df.lr_cnt_pred.values + coefs[2] * df.rf_svd_pred.values 
    print('Optimal AUC')
    print(roc_auc_score(targets, wt_avg))