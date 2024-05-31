from flask import Flask, render_template, request
from src import youtube_data_module as ydt
from src import viz
import os
import logging
import sys

logger = logging.getLogger('app_logger')
handler = logging.StreamHandler(sys.stderr)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

API_KEY = os.getenv('YOUTUBE_API_KEY')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select_video')
def select_video():
    # This page returns search results, when a user hits the 'Search Video' button
    result_dictionary = request.args
    query = result_dictionary['query']
    youtube = ydt.youtubeAPIkey(API_KEY)
    query_result = ydt.youtubeSearchListStatistics(youtube, q=query)

    return render_template(
        'select_video.html',
        query_result=query_result,
        query=query
    )

@app.route('/video_comments')
def video_comments():
    # This page returns a video comment analysis, when a user hits the 'See video comment analysis' button
    video_id = request.args.get('video_id')

    youtube = ydt.youtubeAPIkey(API_KEY)

    logger.info('Getting all comments')
    all_snippets = ydt.get_all_comments(youtube, video_id)
    logger.info('Writing comments to dict')
    comment_dict = ydt.extract_comments(all_snippets)

    image_names = []
    logger.info('Generating wordcloud')
    comment_string = ydt.concat_comments(comment_dict)
    video_title = video_id
    image_names.append(viz.create_wordcloud(comment_string, stopwords=None, video_id=video_id, channel_title=video_title))
    comment_df = ydt.comments_to_df(all_snippets)
    comment_sentiment = ydt.analyze_comment_sentiments(comment_df)
    comment_sentiment2, pos_sent, neg_sent = viz.split_sentiment_pos_neg(comment_sentiment)
    image_names.append(viz.lineplot_cumsum_video_comments(comment_sentiment2, video_id))
    image_names.append(viz.lineplot_cumsum_video_comments_pos_neg(comment_sentiment2, pos_sent, neg_sent, video_id))
    image_names.append(viz.scatterplot_sentiment_likecount(comment_sentiment2, pos_sent, neg_sent, video_id))

    return render_template(
        'video_comments.html',
        image_names=image_names,
    )

if __name__ == '__main__':
    app.run(port=3000, debug=True)
