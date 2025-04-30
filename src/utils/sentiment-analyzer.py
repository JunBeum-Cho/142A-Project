from transformers import pipeline
import pandas as pd


def sentiment_analysis(text_list):
    processed_text_list = [text[:500] for text in text_list]
    classifier = pipeline(
        'sentiment-analysis',
        model='sangrimlee/bert-base-multilingual-cased-nsmc'
    )

    sentiments = classifier(processed_text_list)
    return sentiments


if __name__ == "__main__":
    filename = 'NVIDIA'
    comments = pd.read_csv(f'dataset/comments/{filename}-raw.csv')
    comments_sentiments = sentiment_analysis(comments['content'])
    comments['sentiment'] = [item['score'] if item['label'] == 'positive' else item['score'] * -1 for item in comments_sentiments]

    comments.to_csv(f'dataset/comments/{filename}.csv', index=False, encoding="utf-8-sig")