import requests
import pandas as pd


url = "https://api.opensea.io/api/v1/assets"

# update query string for desired results
querystring = {"order_direction":"desc","offset":"0","limit":"20"}

# add headers/SSL if necessary
response = requests.request("GET", url, params=querystring)
j = response.json()

# create dataframe with one columns of json strings
df = pd.DataFrame.from_dict(j)

# split out json assets to columns
assets_normalized = pd.json_normalize(df.assets)

print(assets_normalized)

assets_normalized.to_csv('assets.csv', index=False)