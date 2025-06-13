from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
import df_merged
import numpy as np


class HybridRecommender:
    def __init__(self, df_reviews):
        """
        初始化混合推荐系统
        """
        self.df = df_reviews
        self.user_ids = list(set(df_reviews['user_id']))
        self.item_ids = list(set(df_reviews['product_id']))

        # 准备Surprise数据集
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_reviews[['user_id', 'product_id', 'rating']], reader)

        # 划分训练集和测试集
        self.trainset, self.testset = train_test_split(data, test_size=0.2)

        # 训练模型
        self.model = KNNWithMeans(k=50, sim_options={'name': 'pearson', 'user_based': False})
        self.model.fit(self.trainset)

        # 计算情感特征权重
        self.calculate_sentiment_weights()

    def calculate_sentiment_weights(self):
        """
        计算情感特征权重
        """
        # 这里简化处理，实际应用中可以使用更复杂的权重计算方法
        self.sentiment_weights = {
            'textblob_polarity': 0.3,
            'vader_compound': 0.4,
            'rating': 0.3
        }

    def predict_rating(self, user_id, product_id):
        """
        预测用户对商品的评分
        """
        # 获取协同过滤预测
        cf_pred = self.model.predict(user_id, product_id).est

        # 获取情感特征
        user_reviews = self.df[(self.df['user_id'] == user_id) & (self.df['product_id'] == product_id)]
        if len(user_reviews) == 0:
            return cf_pred

        avg_polarity = user_reviews['textblob_polarity'].mean()
        avg_vader = user_reviews['vader_compound'].mean()
        avg_rating = user_reviews['rating'].mean()

        # 计算混合评分
        hybrid_score = (self.sentiment_weights['textblob_polarity'] * avg_polarity +
                        self.sentiment_weights['vader_compound'] * avg_vader +
                        self.sentiment_weights['rating'] * avg_rating)

        # 将情感分数映射到1-5评分范围
        hybrid_score = 1 + (hybrid_score + 1) * 2  # 将[-1,1]映射到[1,5]

        # 结合协同过滤和情感分析结果
        final_score = 0.7 * cf_pred + 0.3 * hybrid_score

        return final_score

    def recommend_items(self, user_id, n=5):
        """
        为用户推荐商品
        """
        # 获取用户未评分的商品
        rated_items = set(self.df[self.df['user_id'] == user_id]['product_id'])
        unrated_items = [item for item in self.item_ids if item not in rated_items]

        # 预测评分
        predictions = []
        for item in unrated_items:
            pred = self.predict_rating(user_id, item)
            predictions.append((item, pred))

        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

    def evaluate(self):
        """
        评估模型性能
        """
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        return {'RMSE': rmse, 'MAE': mae}

recommender = HybridRecommender(df_merged)
recommendations = recommender.recommend_items('user1')
print(recommendations)
metrics = recommender.evaluate()
print(metrics)
