import requests
import pandas as pd
import time
from datetime import date, datetime, timedelta


def paginate(max_per_page, limit, url):

    now = datetime.now()
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

    # subset to columns of interest
    final_data = appended_data[[
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

    # convert datetime fields to proper format
    final_data['last_sale.event_timestamp'] = pd.to_datetime(final_data['last_sale.event_timestamp'])
    final_data['last_sale.transaction.timestamp'] = pd.to_datetime(final_data['last_sale.transaction.timestamp'])

    # add script execution datetime as field
    final_data.loc[:,'script_exec_datetime_mt'] = now

    # extract fields from Description
    try:
        final_data[['cv_plotSize_desc','cv_OCdistance_desc','cv_buildHeight_desc','floor_elev_desc']] = final_data.description.str.split(",", expand=True)
        final_data['cv_plotSize_m_sq'] = final_data.cv_plotSize_desc.str.extract('(\d+)')
        final_data['cv_OCdistance_m'] = final_data.cv_OCdistance_desc.str.extract('(\d+)')
        final_data['cv_buildHeight_m'] = final_data.cv_buildHeight_desc.str.extract('(\d+)')
        final_data['cv_floor_elev_m'] = final_data.floor_elev_desc.str.extract('(\d+)')
        final_data.drop(['cv_plotSize_desc','cv_OCdistance_desc','cv_buildHeight_desc','floor_elev_desc'], axis = 1, inplace = True)
    except:
        final_data['cv_plotSize_m_sq'] is null
        final_data['cv_OCdistance_m'] is null
        final_data['cv_buildHeight_m'] is null
        final_data['cv_floor_elev_m'] is null



    print("Dtypes for final data:")
    print(final_data.dtypes)

    final_data.to_csv(f'opensea_asset_data_with_limit={limit}.csv', index=False)

    return appended_data



appended_data = paginate(max_per_page=50, limit=1000, url = "https://api.opensea.io/api/v1/assets?asset_contract_address=0x79986af15539de2db9a5086382daeda917a9cf0c")