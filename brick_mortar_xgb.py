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



### code hur

df_features = read_and_define_scope(file_name = 'train.csv', 
    desired_features = ['SalePrice','1stFlrSF','2ndFlrSF','Neighborhood','HouseStyle','GrLivArea'], 
    target = 'SalePrice',
    viz=True
)