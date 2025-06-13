import pandas as pd
from bs4 import BeautifulSoup
import requests
import re


def scrape_amazon_reviews(product_id, max_pages=5):
    """
    爬取亚马逊商品评论
    """
    base_url = f"https://www.amazon.com/product-reviews/{product_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    reviews = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}/?pageNumber={page}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        for review in soup.find_all('div', {'data-hook': 'review'}):
            try:
                rating = review.find('i', {'data-hook': 'review-star-rating'}).text
                title = review.find('a', {'data-hook': 'review-title'}).text.strip()
                body = review.find('span', {'data-hook': 'review-body'}).text.strip()

                reviews.append({
                    'rating': float(rating.split()[0]),
                    'title': title,
                    'body': body
                })
            except AttributeError:
                continue

    return pd.DataFrame(reviews)

df_reviews = scrape_amazon_reviews("B08N5KWB9H", max_pages=3)
df_reviews.to_csv('amazon_reviews.csv', index=False)
