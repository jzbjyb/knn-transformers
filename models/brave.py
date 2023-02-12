from bs4 import BeautifulSoup
import requests, lxml, json
import time
import random

def get_organic_results(query):
    # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
    params = {
        'q': query, 		# query
        'source': 'web',		# source
        'tf': 'at',				# publish time (by default any time)
        'offset': 0				# pagination (start from page 1)
    }

    # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }

    brave_organic_search_results = []

    html = requests.get('https://search.brave.com/search', headers=headers, params=params)
    soup = BeautifulSoup(html.text, 'lxml')

    for result in soup.select('.snippet'):
        title = result.select_one('.snippet-title').get_text().strip()
        favicon = result.select_one('.favicon').get('src')
        link = result.select_one('.result-header').get('href')
        displayed_link = result.select_one('.snippet-url').get_text().strip().replace('\n', '')

        snippet = result.select_one('.snippet-content .snippet-description , .snippet-description:nth-child(1)').get_text()
        snippet = snippet.strip().split('\n')[-1].lstrip() if snippet else None

        snippet_image = result.select_one('.video-thumb img , .thumb')
        snippet_image = snippet_image.get('src') if snippet_image else None

        rating_and_votes = result.select_one('.ml-10')
        rating = rating_and_votes.get_text().strip().split(' - ')[0] if rating_and_votes else None
        votes = rating_and_votes.get_text().strip().split(' - ')[1] if rating_and_votes else None

        sitelinks_container = result.select('.deep-results-buttons .deep-link')
        sitelinks = None

        if sitelinks_container:
            sitelinks = []
            for sitelink in sitelinks_container:
                sitelinks.append({
                    'title': sitelink.get_text().strip(),
                    'link': sitelink.get('href')
                })

        brave_organic_search_results.append({
            'title': title,
            'favicon': favicon,
            'link': link,
            'displayed_link': displayed_link,
            'snippet': snippet,
            'snippet_image': snippet_image,
            'rating': rating,
            'votes': votes,
            'sitelinks': sitelinks
        })

    return brave_organic_search_results


def get_batch_brave_search_results(queries):
    results = []
    for query in queries:
        results.append(get_organic_results(query))
        time.sleep(1 + random.random())
    return results

if __name__ == "__main__":
    queries = ["hydrogen atomic number",
               "number of spice girls",
               "oxygen atomic number"]

    results = get_batch_brave_search_results(queries)
    print(results)
