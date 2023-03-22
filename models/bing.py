from typing import List, Dict
import time
import os
import json
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
api_url = "http://metis.lti.cs.cmu.edu:8989/bing_search"

def search_bing(query: str, count: int = 10, max_query_per_sec: int = 100, debug: bool = False):
    params = {
        'q': query,
        'mkt': 'en-US',
        'responseFilter':['Webpages'],
        'count': count,
        'safeSearch': 'Off',
        'setLang': 'en-US'
    }
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Pragma': 'no-cache',
        'X-MSEdge-ClientID':'05EFBDF2399564422335AF4538A8655D',
        'User-Agent': "Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko",
        'X-MSEdge-ClientIP': "128.2.211.82"
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        response = response.json()
        if debug:
            print(response)
            input()
        results: List[Dict] = []
        for page in response['webPages']['value']:
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

def search_bing_api(query: str, **kwargs):
    payload = json.dumps({'query': query})
    headers = {'X-Api-Key': 'jzbjybnb', 'Content-Type': 'application/json'}
    try:
        response = requests.request('POST', api_url, headers=headers, data=payload)
        response = response.json()
        results: List[Dict] = []
        for page in response['webPages']['value']:
            result = {
                'url': page['url'],
                'title': page['name'],
                'snippet': page['snippet']
            }
            results.append(result)
        return results
    except Exception as ex:
        raise ex


def search_bing_batch(queries: List[str], count: int = 10):
    results: List[List[Dict]] = []
    for query in queries:
        results.append(search_bing_api(query, count=count))
    return results


if __name__ == '__main__':
    query = "Do people remember Lucille Ball's winemaking as successful?"
    search_bing(query)
