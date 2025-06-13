from flask import Flask, render_template, request
import pandas as pd
from recommender import HybridRecommender

app = Flask(__name__)

# 加载数据
df = pd.read_csv('reviews_with_features.csv')
recommender = HybridRecommender(df)

@app.route('/')
def home():
    return render_template('index.html', users=recommender.user_ids)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id']
    recommendations = recommender.recommend_items(user_id)
    return render_template('recommendations.html',
                         user_id=user_id,
                         recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
