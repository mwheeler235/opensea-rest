from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys

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

    return df_features


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
    one_hot_encoded_nbrhd = pd.get_dummies(df[cat_field])

    df_feat_and_1he_feat = df.join(one_hot_encoded_nbrhd).drop(columns = cat_field, axis = 1)

    return df_feat_and_1he_feat


def training_split(df, target, test_size):

    Y = df[target]
    X = df.drop(columns = target, axis = 1)

    print(f"Features Included: {X.columns}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=82)

    return X_train, X_test, y_train, y_test



def train_model_and_evaluate(X_train, X_test, y_train, y_test):

    # Lambda: regularization parameter that reduces the prediction’s sensitivity to individual observations
    # Gamma: minimum loss reduction required to make a further partition on a leaf node of the tree
    regressor = XGBRegressor(
        n_estimators=100,
        reg_lambda=1,
        gamma=0,
        max_depth=3
    )

    print(regressor)

    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()

    print("...fitting regressor to training data")
    regressor.fit(X_train, y_train)
    print("~~~ Done fitting model ~~~")

    print("XGBoost Feature Importance:")
    print(pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=X_train.columns))

    # call predictions on the test set
    y_pred = regressor.predict(X_test)

    print("Model MSE:", mean_squared_error(y_test, y_pred))

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
    
    df_features = read_and_define_scope(file_name = 'opensea_cryptovoxels_limit=5000_exDTMT=2021-04-27_09_52_41.csv', 
    desired_features = ['last_sale_total_price_adj','cv_plotSize_m_sq','cv_OCdistance_m','cv_buildHeight_m','cv_floor_elev_m','neighborhood'], 
    target = 'last_sale_total_price_adj'
    )

    if encode_type == '1HE':
        df_features_numeric = one_hot_encode(df=df_features, cat_field='neighborhood')
    elif encode_type == 'cat_encode':
        df_features_numeric = categorical_encode(df=df_features, target='last_sale_total_price_adj', cat_fields=['neighborhood'])
    else:
        "encode_type not defined, categorical features not encoded, aborting..."
        sys.exit()

    X_train, X_test, y_train, y_test = training_split(df=df_features_numeric, target='last_sale_total_price_adj', test_size=0.20)

    y_pred = train_model_and_evaluate(X_train, X_test, y_train, y_test)

    joined_results = post_process_and_write_results(y_test=y_test, y_pred=y_pred, X_test=X_test, encode_type=encode_type)


if __name__== "__main__" :

    ## Select preferred encoding method
    #main(encode_type = '1HE')
    main(encode_type = 'cat_encode')