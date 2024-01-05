import lale.lib.aif360
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn
import tpot2
import pandas as pd


def group_race(x):
    if x == "W":
        return 1
    else:
        return 0
    
def calc_weights(X, y, sens_features, privileged_groups, unprivileged_groups):
    ''' Calculate sample weights according to calculationg given in 
           F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.
    ''' 
    (k, v_p), = privileged_groups[0].items()
    (k, v_up), = unprivileged_groups[0].items()

    priv_cond = X[sens_features] == v_p
    priv_cond = priv_cond.to_numpy().flatten()
    unpriv_cond = X[sens_features] == v_up
    unpriv_cond = unpriv_cond.to_numpy().flatten()
    fav_cond = y==1
    unfav_cond = y==0

    # combination of label and privileged/unpriv. groups
    cond_p_fav = np.logical_and(fav_cond, priv_cond)
    cond_p_unfav = np.logical_and(unfav_cond, priv_cond)
    cond_up_fav = np.logical_and(fav_cond, unpriv_cond)
    cond_up_unfav = np.logical_and(unfav_cond, unpriv_cond)

    instance_weights = np.ones(X.shape[0])
    n = X.shape[0]
    n_p = np.sum(instance_weights[priv_cond], dtype=np.float64)
    n_up = np.sum(instance_weights[unpriv_cond], dtype=np.float64)
    n_fav = np.sum(instance_weights[fav_cond], dtype=np.float64)
    n_unfav = np.sum(instance_weights[unfav_cond], dtype=np.float64)

    n_p_fav = np.sum(instance_weights[cond_p_fav], dtype=np.float64)
    n_p_unfav = np.sum(instance_weights[cond_p_unfav],dtype=np.float64)
    n_up_fav = np.sum(instance_weights[cond_up_fav],dtype=np.float64)
    n_up_unfav = np.sum(instance_weights[cond_up_unfav],dtype=np.float64)

    # reweighing weights
    w_p_fav = n_fav*n_p / (n*n_p_fav)
    w_p_unfav = n_unfav*n_p / (n*n_p_unfav)
    w_up_fav = n_fav*n_up / (n*n_up_fav)
    w_up_unfav = n_unfav*n_up / (n*n_up_unfav)

    instance_weights[cond_p_fav] *= w_p_fav
    instance_weights[cond_p_unfav] *= w_p_unfav
    instance_weights[cond_up_fav] *= w_up_fav
    instance_weights[cond_up_unfav] *= w_up_unfav

    return instance_weights

if __name__ == '__main__':
    
    # Set up the dataset; make sure all columns are binary or float valued.
    load_df = getattr(lale.lib.aif360, 'fetch_ricci_df')
    X, y, fairness_info =  load_df()
    l = fairness_info['protected_attributes']
    sens_names = [d['feature'] for d in l]
    X['race'] = X['race'].apply(lambda x: group_race(x))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    preprocessing = tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore")
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)

    fav_label = 'Promotion'
    y_train = pd.Series([1 if y==fav_label else 0 for y in y_train])
    y_test = pd.Series([1 if y==fav_label else 0 for y in y_test])

    
    # Reweighing procedure
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    
    t0 = time.time()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    t1 = time.time()
    print("Without weights: ", t1-t0)
    print(sklearn.metrics.roc_auc_score(y_test, lmod.predict_proba(X_test)[:,1]))

    t0 = time.time()
    sample_weight = calc_weights(X_train, y_train, sens_features=['race'], privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    print(sample_weight)
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train, sample_weight=sample_weight)
    t1 = time.time()
    print("With weights: ", t1-t0)
    print(sklearn.metrics.roc_auc_score(y_test, lmod.predict_proba(X_test)[:,1]))
