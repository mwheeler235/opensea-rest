import requests
import pandas as pd
import time
from datetime import date, datetime, timedelta
import numpy as np
from io import StringIO
import boto3
import sys
import json


def print_pretty(j):
    print(json.dumps(j, indent=2, sort_keys=True, default=str))


def paginate(now, max_per_page, limit, url):

    
    print(f"Script Execution datetime (MT) is {now}")

    # Use offset and max_per_page to paginate
    result_data     = []
    appended_data   = []
    get_data        = True
    api_call        = 0
    result_count    = 0
    offset          = 0

    while get_data == True:

        api_call+=1
        print(f"offset={offset}")

        querystring = {"order_direction":"desc","offset":offset, "limit":limit}

        # GET API response
        response = requests.request("GET", url, params=querystring)
        j = response.json()

        # create dataframe with one columns of json strings
        df = pd.DataFrame.from_dict(j)

        # split out json assets to columns
        results = pd.json_normalize(df.assets)

        print(f'Result preview from offset = {offset}')
        print(results.head())

        result_count = len(results)
        print(f"Records returned in API call {api_call}: ", result_count)

        # error handling: if API call has zero records
        if result_count == 0:
            get_data = False
        # Keep calling API if length of appended data is less than Limit
        elif len(appended_data) < limit-max_per_page:
            #time.sleep(2)

            result_data.append(results)
            # concat list to dataframe
            appended_data = pd.concat(result_data)

            print(f"Cumulative results have {len(appended_data)}")
            offset = offset + max_per_page
            print(f'Adding to result_data, offset set to {str(offset)}')
            get_data = True
        # Final API call
        else:
            try:
                result_data.append(results)
                appended_data = pd.concat(result_data)
                print(f'Adding final data to result_data, cumulative results have {str(len(appended_data))} records. Setting get_data = False.')
            except:
                print(f'No Results to append! Setting get_data = False')

            get_data = False
        
    print(f"Iterations finished. Results have {len(appended_data)} records.")

    # write full column data as well
    #appended_data.to_csv(f'opensea_cryptovoxel_FULL_data_with_limit={limit}_exDTMT={now}.csv')

    # subset to columns of interest
    slim_data = appended_data[[
        'id',
        'name',
        'description',
        'traits',
        'asset_contract.address',
        'asset_contract.asset_contract_type',
        'asset_contract.created_date',
        'asset_contract.name',
        'asset_contract.description',
        'collection.description',
        'collection.name',
        'last_sale.total_price',
        'last_sale.payment_token.symbol',
        'last_sale.event_timestamp',
        'last_sale.transaction.timestamp',
        'last_sale.transaction.to_account.user.username'
    ]]

    return slim_data, limit


def extract_fields(df):

    # drop records with complete nulls
    df = df.dropna(how='all')

    # convert datetime fields to proper format
    df['last_sale.event_timestamp'] = pd.to_datetime(df['last_sale.event_timestamp'])
    df['last_sale.transaction.timestamp'] = pd.to_datetime(df['last_sale.transaction.timestamp'])

    # add script execution datetime as field
    df.loc[:,'script_exec_datetime_mt'] = now

    # Segregate two types of records into two DFs
    #   1. "Odd Records": Find "and near to" then extract first three fields only
    #   2. "Main Records": Extract 4 fields
    df_odd_recs = df.loc[df['description'].str.contains("and near to")]
    df_main_recs = df.loc[~df['description'].str.contains("and near to")]

    #### 1
    # extract fields for the odd df
    try:  
        df_odd_recs[['string1','near_to']] = df_odd_recs.description.str.split("and near to", expand=True)
        df_odd_recs[['cv_plotSize_desc','cv_OCdistance_desc','cv_buildHeight_desc']] = df_odd_recs.string1.str.split(",", expand=True)
        df_odd_recs['cv_plotSize_m_sq'] = df_odd_recs.cv_plotSize_desc.str.extract('(\d+)')
        df_odd_recs['cv_OCdistance_m'] = df_odd_recs.cv_OCdistance_desc.str.extract('(\d+)')
        df_odd_recs['cv_buildHeight_m'] = df_odd_recs.cv_buildHeight_desc.str.extract('(\d+)')

        df_odd_recs.drop(['string1'], axis = 1, inplace = True)
        
        # add addtl fields as NULL for later concat
        df_odd_recs['cv_floor_elev_m'] = np.nan
        df_odd_recs["neighborhood_temp"]= np.nan
        df_odd_recs["neighborhood"]= np.nan
    except:
        df_odd_recs['cv_plotSize_m_sq'] = np.nan
        df_odd_recs['cv_OCdistance_m'] = np.nan
        df_odd_recs['cv_buildHeight_m'] = np.nan
        df_odd_recs['cv_floor_elev_m'] = np.nan
        df_odd_recs["neighborhood_temp"]= np.nan
        df_odd_recs["neighborhood"]= np.nan


    #### 2
    # extract fields for main df
    try:
        df_main_recs[['cv_plotSize_desc','cv_OCdistance_desc','cv_buildHeight_desc','floor_elev_desc']] = df_main_recs.description.str.split(",", expand=True)
        df_main_recs['cv_plotSize_m_sq'] = df_main_recs.cv_plotSize_desc.str.extract('(\d+)')
        df_main_recs['cv_OCdistance_m'] = df_main_recs.cv_OCdistance_desc.str.extract('(\d+)')
        df_main_recs['cv_buildHeight_m'] = df_main_recs.cv_buildHeight_desc.str.extract('(\d+)')
        df_main_recs['cv_floor_elev_m'] = df_main_recs.floor_elev_desc.str.extract('(\d+)')

        # add "near_to" field as NULL
        df_main_recs['near_to'] = np.nan
        
        # Create neighborhood_temp from first item in description
        df_main_recs["neighborhood_temp"]= df_main_recs["cv_plotSize_desc"].str.replace("^.*(?= on )", "")
    except:
        df_main_recs['cv_plotSize_m_sq'] = np.nan
        df_main_recs['cv_OCdistance_m'] = np.nan
        df_main_recs['cv_buildHeight_m'] = np.nan
        df_main_recs['cv_floor_elev_m'] = np.nan
        df_main_recs["neighborhood_temp"]= np.nan
    
    # Extract actual neighborhood from full string (remove " on ")
    try:
        df_main_recs["neighborhood"] = df_main_recs.neighborhood_temp.str.split("on ", expand=True)[1]
    except:
        df_main_recs["neighborhood"] = np.nan


    # stack both DFs
    df = pd.concat([df_main_recs, df_odd_recs], axis=0)
    df.reset_index(drop=True, inplace=True)
    df.drop(['cv_plotSize_desc','cv_OCdistance_desc','cv_buildHeight_desc','floor_elev_desc','neighborhood_temp'], axis = 1, inplace = True)

    print("Dtypes for final data:")
    print(df.dtypes)

    return df


def write_csv_to_s3(bucket, key, df, limit, now):
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'{key}/opensea_cryptovoxels_limit={limit}_exDTMT={now}.csv').put(Body=csv_buffer.getvalue())
 
    params = {
        'Bucket': bucket,
        'Prefix': key
    }
 
    response = client_s3.list_objects_v2(**params)
    print("Objects in Bucket Keys:")
    print_pretty(response['Contents'])


def lambda_handler(event, context):
    print(event)

    # Configure boto session and sts_client for assuming roles 
    #session     = boto3.session.Session(profile_name='mateosanchez')
    session     = boto3.session.Session()
    client_s3   = session.client('s3')
    #sts_client = session.client('sts')

    # REST API limit = 50
    max_per_page = 50

    now = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    now = now.replace(" ", "_")
    today = date.today()
    date_y_m_d = today.strftime("%Y-%m-%d")

    slim_data, limit    = paginate(now=now, max_per_page=max_per_page, limit=150, url = "https://api.opensea.io/api/v1/assets?asset_contract_address=0x79986af15539de2db9a5086382daeda917a9cf0c")
    final_data          = extract_fields(df=slim_data)

    write_csv_to_s3(bucket="opensea-data", key="cryptovoxel_data", df=final_data, limit=limit, now=now)