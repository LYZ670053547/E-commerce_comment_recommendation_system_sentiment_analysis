import pandas as pd

# 生成模拟数据示例
data = {
    'user_id': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3'],
    'product_id': ['prod1', 'prod2', 'prod1', 'prod3', 'prod2', 'prod3'],
    'rating': [5, 4, 3, 5, 2, 4],
    'title': ['Great phone', 'Good but expensive', 'Average performance',
             'Excellent!', 'Disappointed', 'Worth the price'],
    'body': ['I love this phone. The camera is amazing!',
             'Good phone but too expensive for what it offers',
             'The performance is okay, nothing special',
             'Best phone I ever had!',
             'Battery life is terrible',
             'Great value for money']
}

df = pd.DataFrame(data)
print(df)
