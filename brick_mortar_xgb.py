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
import time


def viz_distribution(df, target, title):

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[target])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(title)

    # QQ-plot
    fig = plt.figure()
    res = stats.probplot(df[target], plot=plt)
    plt.show()


def pearson_correlations(df, target):

    pd.set_option('precision',2)
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=.3, bottom=.3)
    sns.heatmap(df.drop([target],axis=1).corr(), square=True)
    plt.suptitle("Pearson Correlation Heatmap")
    plt.show()


def corr_with_target(df, target):
    
    corr_with_sale_price = df.corr()[target].sort_values(ascending=False)
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(bottom=.30)
    corr_with_sale_price.drop(target).plot.bar()
    plt.suptitle("Numeric Predictor Correlation with Target")
    plt.show()


def read_and_define_scope(file_name, desired_features, target, viz):

    df = pd.read_csv(f'./zillow_data/{file_name}')
    df = df.replace(np.nan, None)
    print(f"Raw data has {df['Id'].count()} records.")

    # subset data to ETH and WETH... convert all prices to ETH? apparently they should be 1:1
    # df = df.loc[df['last_sale.payment_token.symbol'].isin(['ETH','WETH'])]
    # print(f"Data subset with symbol in ['ETH', 'WETH'] has {df['Id'].count()} records.")

    # subset columns to target and desired features
    df_features = df[desired_features]

    # remove records where target is NULL (only look at assets that have been sold already)
    df_features = df_features[df_features[target].notnull()]
    print(f"Data with non-null target has {df_features[target].count()} records.")

    if viz==True:
        # viz target distribution
        sns.distplot(df_features[target], fit=norm)
        viz_distribution(df=df_features, target=target, title='Sale Price distribution')

        # viz normalized target distribution
        sns.distplot(np.log1p(df[target]), fit=norm)
        viz_distribution(df=df_features, target=target, title='log(Sale Price+1) distribution')

        # correlation between sale price and numeric features
        pearson_correlations(df=df_features, target=target)

        # get predictor correlations with target BEFORE encoding (too many encoded fields to viz)
        corr_with_target(df=df_features, target=target)
    else:
        pass

    return df_features


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
    params = {"colsample_bytree": stats.uniform(0.7, 0.3), # subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed
            #"colsample_bynode": stats.uniform(0.5, 1),
            "gamma": stats.uniform(0, 2), # defaut=0
            #"lambda": stats.uniform(0.5, 1.5),
            "learning_rate": stats.uniform(0.001, 0.2), # default 0.3 
            "max_depth": randint(2, 6), # default 6; Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
            "n_estimators": randint(100, 250), # default 100
            "subsample": stats.uniform(0.7, 0.3), # Subsample ratio of the training instances. Setting it to 0.5 means randomly sample half of the training data prior to growing trees
            "min_child_weight": stats.uniform(1, 3)
            }

    numFolds    = 5
    n_iter      = 3
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


def post_process_and_write_results(y_test, y_pred, X_test):
    
    y_pred_df = pd.DataFrame(y_pred).rename(columns={0: "predicted_price"})
    y_test_df = pd.DataFrame(y_test).rename(columns={"SalePrice": "actual_price"})
    y_test_df.reset_index(inplace=True, drop=True)

    X_test.reset_index(inplace=True)

    joined_results = y_pred_df.join(y_test_df).join(X_test)

    out_file = f"brick_mortar_xgb_results.csv"

    joined_results.to_csv(out_file, index=False)
    print(f"XGBoost model results written to {out_file}")

    return joined_results


### code hur

df_features = read_and_define_scope(file_name = 'train.csv', 
    desired_features = ['SalePrice','1stFlrSF','2ndFlrSF','Neighborhood','HouseStyle','GrLivArea'], 
    target = 'SalePrice',
    viz=True
)

df_features_numeric = one_hot_encode(df=df_features, cat_field='Neighborhood')
df_features_numeric = one_hot_encode(df=df_features_numeric, cat_field='HouseStyle')

X_train, X_test, y_train, y_test, X = training_split(df=df_features_numeric, target='SalePrice', test_size=0.30)

y_pred = train_model_and_evaluate(X_train, X_test, y_train, y_test, X)

joined_results = post_process_and_write_results(y_test=y_test, y_pred=y_pred, X_test=X_test)