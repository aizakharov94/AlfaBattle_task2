import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold


def smooth_statistic(statistic_set, apply_set, target, features, alpha=30):
    """
    1) statistic_set = X_train
    2) apply_set = X_test
    3) target = string (column name)
    4) features = features for groupby (list of strings)
    """
    new_feature_name = 'smooth_mean_' + '_'.join(features)
    N = statistic_set.shape[0]
    apply_set = apply_set[features]
    statistic_set = statistic_set[features + [target]]
    global_stat_y = np.mean(statistic_set[target])
    means = statistic_set.groupby(features).mean().reset_index()
    means.columns = features + [new_feature_name]
    apply_set = apply_set.merge(right=means, how='left', on=features)
    apply_set[new_feature_name] = (N * apply_set[new_feature_name] + global_stat_y * alpha) / (N + alpha)
    apply_set = apply_set.fillna(global_stat_y)
    return apply_set


def count_statistic_by_folds(train, target, features, kf=None, alpha=30):
    """
    1) train = X_train
    2) target = string (column name)
    3) features = features for groupby (list of strings)
    """
    new_feature_name = 'smooth_mean_' + '_'.join(features)
    train = train[features + [target]]
    target_features = []
    indexes = []
    if len(train[target].unique()) == 2:
        if kf == None:
            skf = StratifiedKFold(n_splits=5, shuffle=True)
        else:
            skf = kf
        for i1, j1 in skf.split(train.drop(target, axis=1), train[target]):
            target_features.append(smooth_statistic(train.iloc[i1], train.iloc[j1], target, features))
            indexes.append(j1)
    else:
        if kf == None:
            skf = KFold(n_splits=5, shuffle=True)
        else:
            skf = kf
        for i1, j1 in skf.split(train.drop(target, axis=1)):
            target_features.append(smooth_statistic(train.iloc[i1], train.iloc[j1], target, features))
            indexes.append(j1)
    return_set = pd.concat(target_features)
    return_set.index = np.concatenate(indexes)
    return return_set[features + [new_feature_name]].sort_index()

def calc_smooth_statistics_features(df, df_test, cat_cols, target_name, kf=None):
    for col in tqdm(cat_cols):
        new_feature_name = 'smooth_mean_' + col
        csbf = count_statistic_by_folds(train=df, target=target_name, features=[col], kf=kf)
        ss = smooth_statistic(statistic_set=df, apply_set=df_test, target=target_name, features=[col])
        df[new_feature_name] = csbf[new_feature_name]
        df_test[new_feature_name] = ss[new_feature_name]
    return df, df_test