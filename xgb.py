from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np


df = pd.read_csv('./cryptovoxel_data/opensea_cryptovoxels_limit=5000_exDTMT=2021-04-27_09_52_41.csv')
df = df.replace(np.nan, None)

print(f"Raw data has {df['id'].count()} records.")

# subset to target and desired features
df_features = df[['last_sale_total_price_adj','cv_plotSize_m_sq','cv_OCdistance_m','cv_buildHeight_m','cv_floor_elev_m','neighborhood']]

# remove records where target is NULL
df_features = df_features[df_features['last_sale_total_price_adj'].notnull()]
print(f"Data with non-null target has {df_features['last_sale_total_price_adj'].count()} records.")

#TODO: subset data to ETH and WETH... or convert all prices to ETH?


#TODO: One hot encode neighborhood

target = ['last_sale_total_price_adj']
target_encode_columns = ['neighborhood']

target_encode_df = df_features[target_encode_columns + target].reset_index().drop(columns = 'index', axis = 1)

target_name = target[0]
target_df = pd.DataFrame()


for embed_col in target_encode_columns:
    val_map = target_encode_df.groupby(embed_col)[target].mean().to_dict()[target_name]
    target_df[embed_col] = target_encode_df[embed_col].map(val_map).values

score_target_drop = df_features.drop(target_encode_columns, axis = 1).reset_index().drop(columns = 'index', axis = 1)

df_features_numeric = pd.concat([score_target_drop, target_df], axis = 1)


Y = df_features_numeric['last_sale_total_price_adj']
X = df_features_numeric[['cv_plotSize_m_sq','cv_OCdistance_m','cv_buildHeight_m','cv_floor_elev_m','neighborhood']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=82)

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
print(pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=X.columns))

# call predictions on the test set
y_pred = regressor.predict(X_test)

print("Model MSE:", mean_squared_error(y_test, y_pred))


# post-process results
y_pred_df = pd.DataFrame(y_pred).rename(columns={0: "predicted_price"})

y_test_df = pd.DataFrame(y_test).rename(columns={"last_sale_total_price_adj": "actual_price"})
y_test_df.reset_index(inplace=True, drop=True)

X_test.reset_index(inplace=True)

joined_results = y_pred_df.join(y_test_df).join(X_test)

joined_results.to_csv("xgb_results.csv", index=False)
print("XGBoost model results written to CSV")