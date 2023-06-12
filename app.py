from flask import Flask, render_template, request
from googleapiclient.discovery import build

app = Flask(__name__)

@app.route('/fake_news_finder', methods=['GET', 'POST'])
def fake_news_detection():
    if request.method == 'POST':
        # Get the keyword and rate limit from the form
        keyword = request.form['keyword']
        rate_limit = int(request.form['rate_limit'])
        
        # Authenticate to YouTube
        api_key = "AIzaSyBPHs1Pq49RrKiW1BIFl2uJHYrwa7cpeyY"

        # Create YouTube service
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Search videos with keyword
        search_response = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=rate_limit
        ).execute()

        # Extract video information from the search response
        videos = []
        for search_result in search_response.get('items', []):
            video_id = search_result['id']['videoId']
            video_title = search_result['snippet']['title']
            channel_title = search_result['snippet']['channelTitle']
            published_at = search_result['snippet']['publishedAt']

            videos.append({
                'video_id': video_id,
                'video_title': video_title,
                'channel_title': channel_title,
                'published_at': published_at
            })

        return render_template('fake_news_finder.html', videos=videos)

    return render_template('fake_news_finder.html', videos=None)

if __name__ == '__main__':
    app.run(debug=True)
