import json
import os 
from pprint import pprint
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
# subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
# endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/bing/v7.0/search"

def bing_search(queries):
    subscription_key = "YOUR_OWN_API_KEY"
    endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"

    for query in queries:
        # Query term(s) to search for. 
        query = "hydrogen atomic number"

        # Construct a request
        mkt = 'en-US'
        params = { 'q': query, 'mkt': mkt, 'responseFilter':['Webpages'], "count": 20}
        headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

        # Call the API
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()

            # print("Headers:")
            # print(response.headers)

            print("JSON Response:")
            pprint(response.json())
        except Exception as ex:
            raise ex
    