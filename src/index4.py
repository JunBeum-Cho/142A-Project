import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt



def sentiment_analysis(text_list):
    processed_text_list = [text[:500] for text in text_list]
    classifier = pipeline(
        'sentiment-analysis',
        model='sangrimlee/bert-base-multilingual-cased-nsmc'
    )

    result = classifier(processed_text_list)
    return result

# 1. 데이터 로드 및 전처리
def main(comment_file, stock_file):
    # 댓글 데이터 로드
    comments = pd.read_csv(comment_file, names=['content', 'timestamp'])[1:1000]
    comments_sentiment = sentiment_analysis(comments['content'])
    print(comments_sentiment)
    comments['sentiment'] = [item['score'] if item['label'] == 'positive' else item['score'] * -1 for item in comments_sentiment]
    # 주식 데이터 로드
    stocks = pd.read_csv(stock_file, names=['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %'])[1:]
    # 댓글 데이터 전처리
    comments['date'] = pd.to_datetime(comments['timestamp'], unit='s').dt.date
    daily_sentiment = comments.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    # 주식 데이터 전처리
    stocks['Date'] = pd.to_datetime(stocks['Date'], format='%m/%d/%Y')
    stocks = stocks.drop(columns=['Vol.'])

    stocks = stocks.sort_values('Date')
    stocks['Price_next_week'] = stocks['Price'].shift(-7)
    stocks = stocks.dropna(subset=['Price_next_week'])

    # 데이터 병합
    merged_data = pd.merge(daily_sentiment, stocks, left_on='date', right_on='Date', how='inner')
    merged_data = merged_data.drop(columns=['date'])

    # 특징 엔지니어링
    features = ['sentiment', 'Open', 'High', 'Low']
    target = 'Price_next_week'
    X = merged_data[features]
    y = merged_data[target]

    print(X)
    

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 훈련 및 평가
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        # 'SVR': SVR(kernel='rbf')
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")

    # 공포 상황 분석
    test_data = X_test.copy()
    test_data['Actual'] = y_test
    for name, y_pred in predictions.items():
        test_data[f'Predicted_{name}'] = y_pred

    fearful_data = test_data[test_data['sentiment'] < -0.5]
    for name in models.keys():
        mse_fear = mean_squared_error(fearful_data['Actual'], fearful_data[f'Predicted_{name}'])
        r2_fear = r2_score(fearful_data['Actual'], fearful_data[f'Predicted_{name}'])
        print(f"{name} (공포 상황) - MSE: {mse_fear:.4f}, R^2: {r2_fear:.4f}")


main('dataset/comments/NVIDIA.csv', 'dataset/stocks/NVIDIA.csv')