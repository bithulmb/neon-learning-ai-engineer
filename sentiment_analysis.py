from transformers import pipeline

from dotenv import load_dotenv

load_dotenv()

class Analyser:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

    def analyse_sentiment(self, text):
        candidate_labels = ["positive", "negative", "neutral"]
        result = self.classifier(text, candidate_labels)
        return result   

def main():
    
    analyzer = Analyser()

    while True:
        text = input('Enter some text (type q to terminate) : ').strip().lower()

        if text == 'q':
            print('Good Luck')
            break

        sentiment = analyzer.analyse_sentiment(text)
        print(sentiment['labels'])
        print(sentiment['scores'])


if __name__ == "__main__":
    main()