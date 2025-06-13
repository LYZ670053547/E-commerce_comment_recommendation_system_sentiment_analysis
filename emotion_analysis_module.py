from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

nltk.download('vader_lexicon')
nltk.download('stopwords')


def analyze_sentiment(text):
    """
    使用TextBlob和VADER进行情感分析
    """
    # TextBlob情感分析
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    # VADER情感分析
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)

    return {
        'textblob_polarity': tb_polarity,
        'textblob_subjectivity': tb_subjectivity,
        'vader_neg': vader_scores['neg'],
        'vader_neu': vader_scores['neu'],
        'vader_pos': vader_scores['pos'],
        'vader_compound': vader_scores['compound']
    }


def preprocess_text(text):
    """
    文本预处理
    """
    # 转换为小写
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def extract_features(df):
    """
    提取文本特征
    """
    # 预处理评论文本
    df['processed_body'] = df['body'].apply(preprocess_text)

    # 应用情感分析
    sentiment_results = df['body'].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df, sentiment_results], axis=1)

    # TF-IDF特征提取
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_features = vectorizer.fit_transform(df['processed_body'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

    return pd.concat([df, tfidf_df], axis=1)

df = pd.read_csv('amazon_reviews.csv')
df_features = extract_features(df)
df_features.to_csv('reviews_with_features.csv', index=False)
