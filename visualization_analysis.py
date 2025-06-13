import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def visualize_sentiment(df):
    """
    可视化情感分析结果
    """
    plt.figure(figsize=(15, 5))

    # 评分分布
    plt.subplot(1, 3, 1)
    sns.countplot(x='rating', data=df)
    plt.title('Rating Distribution')

    # TextBlob情感极性分布
    plt.subplot(1, 3, 2)
    sns.histplot(df['textblob_polarity'], bins=20, kde=True)
    plt.title('TextBlob Polarity Distribution')

    # VADER复合情感分布
    plt.subplot(1, 3, 3)
    sns.histplot(df['vader_compound'], bins=20, kde=True)
    plt.title('VADER Compound Sentiment Distribution')

    plt.tight_layout()
    plt.show()


def visualize_word_cloud(df):
    """
    生成词云
    """

    all_text = ' '.join(df['body'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Frequent Words in Reviews')
    plt.show()

df = pd.read_csv('reviews_with_features.csv')
visualize_sentiment(df)
visualize_word_cloud(df)
