from typing import List, Dict
import time
import os
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
# subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
# endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/bing/v7.0/search"

subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"

def search_bing(query: str, count: int = 10, max_query_per_sec: int = 0.2):
    mkt = 'en-US'
    params = {
        'q': query,
        'mkt': mkt,
        'responseFilter':['Webpages'],
        'count': count
    }
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        results: List[Dict] = []
        for page in response.json()['webPages']['value']:
            result = {
                'url': page['url'],
                'title': page['name'],
                'snippet': page['snippet']
            }
            results.append(result)
        time.sleep(1 / max_query_per_sec)
        return results
    except Exception as ex:
        raise ex

def search_bing_batch(queries: List[str], count: int = 10):
    results: List[List[Dict]] = []
    for query in queries:
        results.append(search_bing(query, count=count))
    return results
