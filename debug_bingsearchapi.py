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

def bing_search(query):
    # subscription_key = "YOUR_OWN_API_KEY"
    subscription_key = ""
    endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"
    
    # Construct a request
    params = {'q': query, 
              'mkt': 'en-US', 
              'responseFilter':['Webpages'], 
              "count": 1,
              "safeSearch": 'Off',
              "setLang": 'en-US'}
    
    headers = {'Ocp-Apim-Subscription-Key': subscription_key ,
               'Pragma': 'no-cache',
               'X-MSEdge-ClientID':'05EFBDF2399564422335AF4538A8655D',
               'User-Agent': "Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko",
               'X-MSEdge-ClientIP': "128.2.211.82"}

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        print("Headers:")
        print(response.headers)

        print("JSON Response:")
        pprint(response.json())
    except Exception as ex:
        raise ex

if __name__ == "__main__":
    bing_search("Do people remember Lucille Ball's winemaking as successful?")