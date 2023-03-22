import json
import os 
from pprint import pprint
import requests
from ratelimit import limits, sleep_and_retry

from flask import Flask, request, jsonify
from diskcache import Cache

app = Flask(__name__)
cache = Cache('bing_search_cache')

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
# subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
# endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/bing/v7.0/search"
subscription_key = open('bing_api_key.txt').read().strip()

@sleep_and_retry
@limits(calls=50, period=1)
def bing_search(query):
    if query in cache:
        print('cache_hit:', query)
        return cache[query]
    else:
        endpoint = "https://api.bing.microsoft.com/" + "v7.0/search"
        
        # Construct a request
        params = {'q': query, 
                'mkt': 'en-US', 
                'responseFilter':['Webpages'], 
                "count": 50,
                "safeSearch": 'Off',
                "setLang": 'en-US'}
        
        headers = {'Ocp-Apim-Subscription-Key': subscription_key ,
                'Pragma': 'no-cache',
                'X-MSEdge-ClientID':'05EFBDF2399564422335AF4538A8655D',
                'User-Agent': "Mozilla/5.0 (Windows NT 6.3; WOW64; Tridentz/7.0; Touch; rv:11.0) like Gecko",
                'X-MSEdge-ClientIP': "128.2.211.82"}

        # Call the API
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            cache[query] = result
            return(result)
        except Exception as ex:
            print(ex)
            return None

@app.route('/bing_search', methods=['POST'])
def bing_request():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    if auth == 'jzbjybnb':
        data = request.get_json()
        result = bing_search(data['query'])
        if result is None:
            return jsonify({"message": "ERROR: Request Failed"}), 404
        else:
            return jsonify(result), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989)

