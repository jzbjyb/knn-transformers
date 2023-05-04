from typing import List, Dict
import time
import os
import json
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY'] if 'BING_SEARCH_V7_SUBSCRIPTION_KEY' in os.environ else None
endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"
api_url = "http://127.0.0.1:8989/bing_search"  # "http://metis.lti.cs.cmu.edu:8989/bing_search"

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

def search_bing_api(
        query: str,
        only_domain: str = None,
        exclude_domains: List[str] = [],
        count: int = 10,
    ):
    # prepare request
    data = {'query': query}
    if only_domain:
        data['+domain'] = only_domain
    elif exclude_domains:
        data['-domain'] = ','.join(exclude_domains)
    data = json.dumps(data)
    headers = {'X-Api-Key': 'jzbjybnb', 'Content-Type': 'application/json'}
    try:
        # sent request
        response = requests.request('POST', api_url, headers=headers, data=data)
        response = response.json()
        results: List[Dict] = []
        # collect results
        if 'webPages' in response and 'value' in response['webPages']:
            for page in response['webPages']['value']:
                result = {
                    'url': page['url'],
                    'title': page['name'],
                    'snippet': page['snippet']
                }
                exclude = False
                if exclude_domains:
                    for d in exclude_domains:
                        if d in page['url']:
                            exclude = True
                            break
                if not exclude:
                    results.append(result)
        return results
    except Exception as ex:
        raise ex


def search_bing_batch(queries: List[str], **kwargs):
    results: List[List[Dict]] = []
    for query in queries:
        results.append(search_bing_api(query, **kwargs))
    return results


if __name__ == '__main__':
    query = "Do people remember Lucille Ball's winemaking as successful?"
    search_bing(query)
