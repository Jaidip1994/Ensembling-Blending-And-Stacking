import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def run_train(fold):
    df = pd.read_csv('../input/train_folds.csv')
    df.review = df.review.apply(str)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer(max_features=1000)
    tfv.fit(df_train.review.values)

    X_train = tfv.transform(df_train.review.values)
    X_valid = tfv.transform(df_valid.review.values)

    y_train = df_train.sentiment.values
    y_valid = df_valid.sentiment.values

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    pred = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    print(f'Fold= {fold}, AUC = {auc}')

    df_valid.loc[:, 'lr_pred'] = pred
    return df_valid[['id', 'sentiment', 'kfold', 'lr_pred']]

if __name__ == "__main__":
    dfs =  []
    for j in range(5):
        temp_df = run_train(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv('../model_pred/lr.csv', index=False)