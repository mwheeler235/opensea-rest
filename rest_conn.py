import requests
import pandas as pd
import time


url = "https://api.opensea.io/api/v1/assets?asset_contract_address=0x79986af15539de2db9a5086382daeda917a9cf0c"



# Use offset and max_per_page to paginate
result_data     = []
get_data        = True
offset          = 0
max_per_page    = 50
limit           = 150

while get_data == True:

    print(f"offset={offset}")

    # update query string for desired results
    querystring = {"order_direction":"desc","offset":offset, "limit":limit}

    # add headers/SSL if necessary
    response = requests.request("GET", url, params=querystring)
    j = response.json()

    # create dataframe with one columns of json strings
    df = pd.DataFrame.from_dict(j)

    # split out json assets to columns
    results = pd.json_normalize(df.assets)

    print(f'Result preview from offest = {offset}')
    print(results.head())

    result_count = len(results)
    print("Records returned: ", result_count)

    if result_count == 0:
        get_data = False
    elif offset != limit:
        result_data.append(results)
        offset = offset + max_per_page
        print(f'Adding to result_data, offset set to {str(offset)}')
        get_data = True
    else:
        try:
            result_data.append(results)
            print(f'Adding to result_data, result set was {str(len(result_data))}, so setting get_data = False')
        except:
            print(f'No Results to append! Setting get_data = False')

        get_data = False

    print(result_data)
    #result_data = pd.concat(result_data)
    #print(f'Final count for all rows: {len(result_data)}')

    #result_data.to_csv('assets.csv', index=False)