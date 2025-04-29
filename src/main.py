from transformers import pipeline


def sentiment_analysis(text_list):
    classifier = pipeline(
        'sentiment-analysis',
        model='sangrimlee/bert-base-multilingual-cased-nsmc'
    )

    result = classifier(text_list)
    print(result)


if __name__ == "__main__":
    text_list = []
    sentiment_analysis(text_list)