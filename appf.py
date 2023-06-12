import cv2
import imageio
import numpy as np
from flask import Flask, jsonify, render_template, request
from googleapiclient.discovery import build
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('front.html')

# Route for the fake_news_finder page
@app.route('/fake_news_finder/', methods=['GET', 'POST'])
def fake_news_finder():
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

# Route for the fake_video_detector page
@app.route('/fake_video_detector/', methods=['GET', 'POST'])
def fake_video_detector():
    if request.method == 'POST':
        if 'video-file' in request.files:
            video_file = request.files['video-file']
            temp_file_path = 'temp_video.mp4'
            video_file.save(temp_file_path)

            duration = calculate_video_duration(temp_file_path)
            predictions = process_video(temp_file_path, model)

            if predictions:
                classification, confidence = classify_video(predictions)
                result = {
                    'classification': classification,
                    'confidence': confidence,
                    'duration': duration
                }
                return jsonify(result)  # Return the result as JSON

            else:
                return jsonify({'error': 'Error occurred during video processing.'})

    return render_template('fake_video_detector.html')

# Route for the working page
@app.route('/working/')
def working():
    return render_template('working.html')

# Route for the privacy policy page
@app.route('/privacy_policy/')
def privacy_policy():
    return render_template('privacy_policy.html')

def calculate_video_duration(video_path):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return None

    duration = video.get_meta_data()['duration']
    video.close()

    return duration

def process_video(video_path, model):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return []

    frame_count = len(video)
    frame_height, frame_width = video.get_meta_data()['source_size'][:2]

    print(f"Video info: {frame_count} frames, {frame_width}x{frame_height} resolution")

    predictions = []

    try:
        for i, frame in enumerate(video):
            # Resize and preprocess the frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)

            # Make a prediction
            prediction = model.predict(frame)
            predictions.append(prediction[0][0])  # Append the prediction result (fake score)

    except Exception as e:
        print(f"Error occurred during video processing. {e}")
        return []

    return predictions

def classify_video(predictions, threshold=0.5):
    if not predictions:
        return 'unknown', 0

    fake_count = sum(1 for score in predictions if score > threshold)
    real_count = len(predictions) - fake_count

    fake_score = fake_count / len(predictions)
    real_score = real_count / len(predictions)

    if fake_score > real_score:
        return 'fake', fake_score
    else:
        return 'real', real_score

if __name__ == '__main__':
    # Load the pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

    app.run(debug=True)
