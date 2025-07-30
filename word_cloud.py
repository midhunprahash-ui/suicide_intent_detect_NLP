import pandas as pd
import re
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df = pd.read_csv('final.csv')


def preprocess_text(text):
    """Cleans text by removing punctuation and converting to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


df['cleaned_text'] = df['user_text'].apply(preprocess_text)


words = ' '.join(df['cleaned_text']).split()


stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'im'
])

filtered_words = [word for word in words if word not in stop_words]


positive_words = []
negative_words = []


for word in filtered_words:
    analysis = TextBlob(word)
    if analysis.sentiment.polarity > 0:
        positive_words.append(word)
    elif analysis.sentiment.polarity < 0:
        negative_words.append(word)


positive_word_counts = Counter(positive_words)
if positive_word_counts:

    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_word_counts)
    

    plt.figure(figsize=(10, 5))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Words Word Cloud')
    plt.savefig('positive_word_cloud.png')
    print("Positive word cloud generated and saved as 'positive_word_cloud.png'.")

negative_word_counts = Counter(negative_words)
if negative_word_counts:

    negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(negative_word_counts)
    

    plt.figure(figsize=(10, 5))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Words Word Cloud')
    plt.savefig('negative_word_cloud.png')
    print("Negative word cloud generated and saved as 'negative_word_cloud.png'.")