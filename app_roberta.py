# -*- coding: utf-8 -*-
"""app_roberta.ipynb

## Analysis of Public Perception of Women's Participation in Football during Euro 2024
# Introduction

The importance of equal opportunities for women in sports cannot be overstated. Historically, sports have been a male-dominated arena, with women often facing significant barriers to participation, recognition, and equal treatment. However, the landscape is gradually changing, and events like the Euro 2024 are prime examples of platforms where women athletes are increasingly showcasing their talent and dedication.

# 🚩Analyzing public perception of women's participation in football during major events like Euro 2024 is crucial for several reasons:

✅Understanding Public Sentiment: Gauging how the public perceives women's participation helps stakeholders understand the level of support or opposition. This can influence policies, sponsorship, and media coverage.

✅Identifying Areas for Improvement: Sentiment analysis can highlight specific areas where women's participation is either praised or criticized, allowing organizations to address these issues effectively.

✅Promoting Equality: By continuously monitoring and analyzing public opinion, we can promote equality in sports, ensuring that women receive the recognition and opportunities they deserve.

✅Supporting Decision Making: Organizations, advertisers, and policy-makers can use these insights to make informed decisions that support and promote women's sports.

# 🚩Purpose of the Analysis

The purpose of this analysis is to understand how the participation of women in football during Euro 2024 is perceived. Through sentiment analysis of posts and comments on Reddit, we aim to identify positive, negative, and neutral opinions on this topic. This information can be useful for sports organizations, journalists, and analysts who wish to understand public perception and make informed decisions.

1. APIs Used

✅PRAW (Python Reddit API Wrapper): To access posts and comments on Reddit.
✅Transformers by Hugging Face: To use pre-trained sentiment analysis models.
✅NLTK (Natural Language Toolkit): For sentence tokenization.
✅Plotly: For data visualization.

2. Model Used (Pipelines)

We used the cardiffnlp/twitter-roberta-base-sentiment sentiment analysis model provided by Hugging Face. This model is optimized for analyzing sentiments in short texts, such as social media posts and comments.

## 1.Import Libraries and Configure Models
"""

import pandas as pd
import praw
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.notebook import tqdm
from transformers import pipeline, AutoTokenizer
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import numpy as np
from collections import Counter

#nltk.download('punkt')

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

"""## 2.Functions for Text Processing and Sentiment Analysis"""

def split_text_into_chunks(text, tokenizer, max_length=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length + 2 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def analyze_text_chunks(text):
    chunks = split_text_into_chunks(text, tokenizer, max_length=tokenizer.model_max_length)
    return [sentiment_pipeline(chunk)[0] for chunk in chunks if chunk.strip()]

def aggregate_sentiments(sentiments):
    if not sentiments:
        return {'label': 'neutral', 'score': 0.0}
    avg_score = np.mean([sentiment['score'] for sentiment in sentiments])
    labels = [sentiment['label'] for sentiment in sentiments]
    label = Counter(labels).most_common(1)[0][0]
    return {'label': label, 'score': avg_score}

label_mapping = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}

"""## 3.Function to Fetch Data from Reddit"""

def fetch_reddit_posts_and_comments():
    reddit = praw.Reddit(
        client_id='ZC5eI8EbdWOlcTQU1C7cCg',
        client_secret='a1MvR2C_syEBqKPWC75HRb-li28jSQ',
        user_agent='python:praw:example_app:v1.0 (by /u/Suspicious_Sport2182)'
    )
    subreddit = reddit.subreddit('euro2024')
    query = "women"
    posts_and_comments = []
    for submission in subreddit.search(query, limit=1000):
        try:
            posts_and_comments.append(submission.title + " " + submission.selftext)
            submission.comments.replace_more(limit=0)
            posts_and_comments.extend([comment.body for comment in submission.comments.list()])
        except Exception as e:
            print(f"Error fetching submission or comments: {e}")
    return posts_and_comments

"""## 4.Processing and Sentiment Analysis"""

texts = fetch_reddit_posts_and_comments()

texts_sentiment = []
error_count = 0

def process_text(text):
    try:
        sentiments = analyze_text_chunks(text)
        aggregated_sentiment = aggregate_sentiments(sentiments)
        return {
            'text': text,
            'sentiment_label': label_mapping.get(aggregated_sentiment['label'], 'neutral'),
            'sentiment_score': aggregated_sentiment['score']
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_text, texts), total=len(texts), desc="Analyzing sentiments"))

texts_sentiment = [result for result in results if result is not None]
error_count = len([result for result in results if result is None])
df = pd.DataFrame(texts_sentiment)

"""## 6.Calculate Loss Rate and Display Results"""

total_texts = len(texts)
processed_texts = total_texts - error_count
loss_rate = (error_count / total_texts) * 100

print(f"Total posts and comments analyzed: {total_texts}")
print(f"Total texts processed without errors: {processed_texts}")
print(f"Loss rate: {loss_rate:.2f}%")

"""## 7. Data Visualization

Bar Chart
"""

sentiment_counts = df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment', 'count']
fig_bar = px.bar(sentiment_counts, x='sentiment', y='count',
                 labels={'sentiment': 'Sentiment', 'count': 'Count'},
                 title='Distribution of Sentiments in Posts 💻',
                 color='sentiment',
                 color_discrete_sequence=px.colors.qualitative.Set3)
fig_bar.show()

"""Scatter Plot"""

fig_scatter = px.scatter(df, x='sentiment_score', y='text', color='sentiment_label',
                         title='Sentiment Scores by Text',
                         labels={'sentiment_score': 'Score', 'sentiment_label': 'Sentiment'},
                         hover_data=['text'],
                         color_discrete_sequence=px.colors.qualitative.Set1)
fig_scatter.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
fig_scatter.show()

# Diagrama de dispersión para visualizar los scores de los sentimientos
fig_scatter = px.scatter(df, x='sentiment_score', y='sentiment_label', color='sentiment_label',
                         title='Sentiment Scores by Text',
                         labels={'sentiment_score': 'Score', 'sentiment_label': 'Sentiment'},
                         hover_data=['text'],
                         color_discrete_sequence=px.colors.qualitative.Set1)
fig_scatter.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
fig_scatter.update_layout(yaxis={'categoryorder':'total descending'})
fig_scatter.show()

"""Pie Chart"""

fig_pie = px.pie(df, names='sentiment_label', title='Proportion of Sentiments',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
fig_pie.show()

"""Box Plot"""

fig_box = px.box(df, x='sentiment_label', y='sentiment_score', color='sentiment_label',
                 title='Distribution of Sentiment Scores',
                 labels={'sentiment_label': 'Sentiment', 'sentiment_score': 'Score'},
                 color_discrete_sequence=px.colors.qualitative.Vivid)
fig_box.show()

"""## 8. Conclusions

Through this analysis, we can draw several key conclusions about the public perception of women's participation in football during Euro 2024:

Distribution of Sentiments: Most of the analyzed texts exhibit a neutral sentiment, followed by positive and then negative sentiments.

Proportion of Sentiments: The pie chart clearly shows the proportion of each sentiment type, indicating that the perception is mostly neutral or positive.

Sentiment Scores: The box plot shows the distribution of sentiment scores, indicating variability within each sentiment category.

This analysis provides a clear view of how the participation of women in football during Euro 2024 is perceived on the Reddit platform, helping guide future strategies and communications in the sports and social spheres.
"""