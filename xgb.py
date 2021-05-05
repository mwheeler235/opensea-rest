from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys
from scipy import stats
from scipy.stats import randint, norm
import matplotlib.pyplot as plt
import seaborn as sns


def viz_distribution(df, target, title):

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[target])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(title)

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(df[target], plot=plt)
    plt.show()


def pearson_correlations(df, target):

    pd.set_option('precision',2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop([target],axis=1).corr(), square=True)
    plt.suptitle("Pearson Correlation Heatmap")
    plt.show()


def read_and_define_scope(file_name, desired_features, target):

    df = pd.read_csv(f'./cryptovoxel_data/{file_name}')
    df = df.replace(np.nan, None)
    print(f"Raw data has {df['id'].count()} records.")

    # subset data to ETH and WETH... convert all prices to ETH? apparently they should be 1:1
    df = df.loc[df['last_sale.payment_token.symbol'].isin(['ETH','WETH'])]
    print(f"Data subset with symbol in ['ETH', 'WETH'] has {df['id'].count()} records.")

    # subset columns to target and desired features
    df_features = df[desired_features]

    # remove records where target is NULL
    df_features = df_features[df_features[target].notnull()]
    print(f"Data with non-null target has {df_features[target].count()} records.")

    # viz target distribution
    sns.distplot(df_features[target], fit=norm)
    viz_distribution(df=df_features, target=target, title='Sale Price distribution')

    # viz normalized target distribution
    sns.distplot(np.log1p(df[target]), fit=norm)
    viz_distribution(df=df_features, target=target, title='log(Sale Price+1) distribution')

    # correlation between sale price and numeric features
    pearson_correlations(df=df_features, target=target)

    return df_features


def normalize_target():
    #TODO
    pass


def categorical_encode(df, target, cat_fields):

    target = [target]
    target_encode_columns = cat_fields

    target_encode_df = df[target_encode_columns + target].reset_index().drop(columns = 'index', axis = 1)

    target_name = target[0]
    target_df = pd.DataFrame()

    for embed_col in target_encode_columns:
        val_map = target_encode_df.groupby(embed_col)[target].mean().to_dict()[target_name]
        target_df[embed_col] = target_encode_df[embed_col].map(val_map).values

    score_target_drop = df.drop(target_encode_columns, axis = 1).reset_index().drop(columns = 'index', axis = 1)

    df_features_numeric = pd.concat([score_target_drop, target_df], axis = 1)

    return df_features_numeric


def one_hot_encode(df, cat_field):
    one_hot_encoded_fields = pd.get_dummies(df[cat_field])

    df_feat_and_1he_feat = df.join(one_hot_encoded_fields).drop(columns = cat_field, axis = 1)

    return df_feat_and_1he_feat


def training_split(df, target, test_size):

    Y = df[target]
    X = df.drop(columns = target, axis = 1)

    print(f"Features Included: {X.columns}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=82)

    return X_train, X_test, y_train, y_test, X



def train_model_and_evaluate(X_train, X_test, y_train, y_test, X):

    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()

    # Lambda: regularization parameter that reduces the prediction’s sensitivity to individual observations
    # Gamma: minimum loss reduction required to make a further partition on a leaf node of the tree
   
    # Hyperparameter ranges:
    params = {"colsample_bytree": stats.uniform(0.7, 0.3),
            "colsample_bynode": stats.uniform(0.5, 1),
            "gamma": stats.uniform(0, 0.5),
            "learning_rate": stats.uniform(0.001, 0.2), # default 0.1 
            "max_depth": randint(2, 4), # default 3
            "n_estimators": randint(100, 250), # default 100
            "subsample": stats.uniform(0.6, 0.4),
            "min_child_weight": stats.uniform(1, 3)
            }

    numFolds    = 5
    n_iter      = 4
    folds = KFold(n_splits=numFolds, shuffle=True)
    
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=29)

    print(xgb_model)

    xgb_search = RandomizedSearchCV(xgb_model, 
    param_distributions=params, 
    random_state=42, 
    n_iter=n_iter, 
    cv=folds, 
    verbose=1, 
    n_jobs=2, # parallel processing
    return_train_score=True,
    refit=True # this selects the best estimator for retraining
    )

    print(f"...tuning models with {n_iter} iterations")
    xgb_search.fit(X_train, y_train)
    print("~~~ Done tuning models ~~~")

    # print("XGBoost Feature Importance:")
    # print(pd.DataFrame(xgb_search.feature_importances_.reshape(1, -1), columns=X_train.columns))

    print("\n The best estimator across all searched params:\n", xgb_search.best_estimator_)
    print("\n The best score across ALL searched params:\n", xgb_search.best_score_)
    print("\n The best parameters across ALL searched params:\n", xgb_search.best_params_)

    # call predictions on the test set
    y_pred = xgb_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('Model MSE:', mse)
    print('Model RMSE:', rmse)

    print('Explained Variance Score', explained_variance_score(y_pred, y_test))

    return y_pred


def post_process_and_write_results(y_test, y_pred, X_test, encode_type):
    
    y_pred_df = pd.DataFrame(y_pred).rename(columns={0: "predicted_price"})
    y_test_df = pd.DataFrame(y_test).rename(columns={"last_sale_total_price_adj": "actual_price"})
    y_test_df.reset_index(inplace=True, drop=True)

    X_test.reset_index(inplace=True)

    joined_results = y_pred_df.join(y_test_df).join(X_test)

    out_file = f"xgb_results_{encode_type}.csv"

    joined_results.to_csv(out_file, index=False)
    print(f"XGBoost model results written to {out_file}")

    return joined_results


def main(encode_type):
    
    df_features = read_and_define_scope(file_name = 'opensea_cryptovoxels_limit=6000_exDTMT=2021-05-04_09_45_55.csv', 
    desired_features = ['last_sale_total_price_adj','cv_plotSize_m_sq','cv_OCdistance_m','cv_buildHeight_m','cv_floor_elev_m','neighborhood','near_to'], 
    target = 'last_sale_total_price_adj'
    )

    sys.exit()

    if encode_type == '1HE':
        df_features_numeric = one_hot_encode(df=df_features, cat_field='neighborhood')
        df_features_numeric = one_hot_encode(df=df_features_numeric, cat_field='near_to')
    elif encode_type == 'cat_encode':
        df_features_numeric = categorical_encode(df=df_features, target='last_sale_total_price_adj', cat_fields=['neighborhood'])
        df_features_numeric = categorical_encode(df=df_features_numeric, target='last_sale_total_price_adj', cat_fields=['near_to'])
    else:
        "encode_type not defined, categorical features not encoded, aborting..."
        sys.exit()

    X_train, X_test, y_train, y_test, X = training_split(df=df_features_numeric, target='last_sale_total_price_adj', test_size=0.20)

    y_pred = train_model_and_evaluate(X_train, X_test, y_train, y_test, X)

    joined_results = post_process_and_write_results(y_test=y_test, y_pred=y_pred, X_test=X_test, encode_type=encode_type)


if __name__== "__main__" :

    ## Select preferred encoding method
    main(encode_type = '1HE')
    #main(encode_type = 'cat_encode')