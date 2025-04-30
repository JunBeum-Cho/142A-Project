import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime


URL = 'https://wts-cert-api.tossinvest.com/api/v3/comments'
HEADERS = {
  'Connection': 'keep-alive',
  'Origin': 'https://tossinvest.com',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
  'accept': 'application/json',
  'content-type': 'application/json',
}

def crawl_comments(stock_code: str):
  last_comment = None
  all_comments = []

  for i in tqdm(range(10000)):
    data = {
      "commentId": last_comment,
      "subjectId": stock_code,
      "subjectType": "STOCK",
      "commentSortType": "RECENT"
    }

    response = requests.post(URL, headers=HEADERS,json=data).json()
    comments = response.get('result').get('comments').get('body')
    for c in comments:
      content: str = (c.get('title') or '') + c.get('message')
      updated_at = c.get('updatedAt')
      
      if len(content) < 5:
        continue

      all_comments.append({
        "content": content.replace('\n', ' '),
        "timestamp": int(datetime.fromisoformat(updated_at).timestamp())
      })

    last_comment = comments[-1].get('id')
    print(last_comment)
      

  df = pd.DataFrame(all_comments)
  df.to_csv(f'{stock_code}.csv', index=False, encoding="utf-8-sig")
  print(f"Saved {len(all_comments)} comments to {stock_code}")
  
if __name__ == "__main__":
  # TSLA_CODE = 'US20100629001'
  NVIDIA_CODE = 'US19990122001'
  PALANTIR_CODE = 'US20200930014'

  crawl_comments(NVIDIA_CODE)