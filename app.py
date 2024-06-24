from urllib.error import HTTPError
from flask import Flask, render_template, request, Response
from src import youtube_data_module as ydt
from src import viz
import os
import re
import pandas as pd
import logging
import sys
import google.generativeai as genai
import pathlib
import textwrap
from IPython.display import display, Markdown
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger('app_logger')
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

API_KEY = os.getenv('YOUTUBE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select_video')
def select_video():
    try:
        result_dictionary = request.args
        query = result_dictionary['query']
        youtube = ydt.youtubeAPIkey(API_KEY)
        query_result = ydt.youtubeSearchListStatistics(youtube, q=query)
        return render_template('select_video.html', query_result=query_result, query=query)
    except Exception as e:
        logger.error(f"Error selecting video: {str(e)}")
        return render_template('error.html', error_message="An error occurred while selecting the video. Please try again later.")

@app.route('/video_comments')
def video_comments():
    try:
        video_id = request.args.get('video_id')
        negative_Average = 0
        youtube = ydt.youtubeAPIkey(API_KEY)
        logger.info('Getting all comments')
        all_snippets = ydt.get_all_comments(youtube, video_id)
        logger.info('Writing comments to dict')
        comment_dict = ydt.extract_comments(all_snippets)
        image_names = []
        logger.info('Generating wordcloud')
        comment_string = ydt.concat_comments(comment_dict)
        video_title = get_video_title(video_id=video_id)
        image_names.append(viz.create_wordcloud(comment_string, stopwords=None, video_id=video_id, channel_title=video_title))
        comment_df = ydt.comments_to_df(all_snippets)
        comment_sentiment = analyze_comment_sentiment(comment_df)
        comment_sentiment2, pos_sent, neg_sent = viz.split_sentiment_pos_neg(comment_sentiment)
        image_names.append(viz.lineplot_cumsum_video_comments(comment_sentiment2, video_id))
        image_names.append(viz.lineplot_cumsum_video_comments_pos_neg(comment_sentiment2, pos_sent, neg_sent, video_id))
        image_names.append(viz.scatterplot_sentiment_likecount(comment_sentiment2, pos_sent, neg_sent, video_id))
        
        # Calculate average negative sentiment
        negative_Sentiment = comment_sentiment['pos'].mean()
        percentage = (negative_Sentiment / 4.0) * 100
        negative_Average = "{:.2f}%".format(percentage)

        generatedText = ''
        # if negative_Sentiment > 0.10:
        #     model = genai.GenerativeModel('gemini-pro')
        #     response = model.generate_content(f"Suggest me a one video title related to this {video_title} in which I can get more positive comments")
        #     generatedText += response.text if to_markdown(response.text) else "No response generated."

        return render_template('video_comments.html', image_names=image_names, generatedText=generatedText, negative_Average=negative_Average)

    except HTTPError as e:
        error_message = f"An HTTP error occurred: {e}"
        logger.error(error_message)
        return render_template('error.html')

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        return render_template('error.html', error_message=error_message)

def to_markdown(text):
    text = text.replace('**', ' \n')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def analyze_comment_sentiment(comment_df):
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = {'neg': [], 'neu': [], 'pos': [], 'compound': []}
        for comment in comment_df['text_original']:
            vs = analyzer.polarity_scores(comment)
            for k, v in vs.items():
                sentiment[k].append(v)
        sentiment_df = pd.DataFrame(data=sentiment)
        comment_sentiment = pd.concat([comment_df.reset_index(), sentiment_df], axis=1)
        return comment_sentiment
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return pd.DataFrame()

def get_video_title(video_id):
    try:
        youtube = ydt.youtubeAPIkey(API_KEY)
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            return response['items'][0]['snippet']['title']
        else:
            return "Video not found"
    except Exception as e:
        logger.error(f"Error getting video title: {str(e)}")
        return "Unknown Title"

if __name__ == '__main__':
    app.run(port=3000, debug=True)