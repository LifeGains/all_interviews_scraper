from googleapiclient.discovery import build
import os
import pandas as pd
from dotenv import load_dotenv
import requests
import isodate
from pyairtable import Api
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
from googleapiclient.discovery import build
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set Python Debug Console to current cwd
# Get the directory name of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Set the current working directory to the directory of the current file
os.chdir(current_file_directory)

# Path
dotenv_path = 'C:\\Users\\kevin\\Google Drive\\My Drive\\Github\\all-api-keys\\.env'
load_dotenv(dotenv_path)

# Load api key
# Set the path to the service account credentials
google_json_path = 'C:\\Users\\kevin\\Google Drive\\My Drive\\Github\\all-api-keys\\youtube_api_private_key_service_account.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_json_path

airtable_api_key = os.getenv("AIRTABLE_API")
api = Api(airtable_api_key)

# Build the YouTube service without passing the developerKey
youtube = build('youtube', 'v3')

def search_videos_advanced(query, max_results_per_request=50, total_max_results=200, video_length='long'):
    videos_data = []
    next_page_token = None
    
    while len(videos_data) < total_max_results:
        # Make a request to the YouTube API to search for videos
        search_request = youtube.search().list(
            part='snippet',
            q=query,
            maxResults=max_results_per_request,
            type='video',
            videoDuration=video_length,
            pageToken=next_page_token
        )
        search_response = search_request.execute()

        # Collect video IDs from the search response
        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

        # Make a request to the YouTube API to get video details
        videos_request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(video_ids)
        )
        videos_response = videos_request.execute()

        # Parse the response and collect data
        for item in videos_response.get('items', []):
            # Extract all data from the snippet and statistics
            snippet_data = item['snippet']
            statistics_data = item['statistics']

            video_data = {
                'PublishedAt': snippet_data.get('publishedAt'),
                'ChannelId': snippet_data.get('channelId'),
                'Title': snippet_data.get('title'),
                'Description': snippet_data.get('description'),
                'Thumbnails': snippet_data.get('thumbnails', {}).get('default', {}).get('url'),  # You can choose which thumbnail size you want
                'ChannelTitle': snippet_data.get('channelTitle'),
                'LiveBroadcastContent': snippet_data.get('liveBroadcastContent'),
                'PublishTime': snippet_data.get('publishTime'),
                'VideoID': item['id'],
                'URL': f'https://www.youtube.com/watch?v={item["id"]}',
                'Duration': duration_to_minutes(item['contentDetails']['duration']),
                'ViewCount': statistics_data.get('viewCount'),
                'LikeCount': statistics_data.get('likeCount'),
                'DislikeCount': statistics_data.get('dislikeCount'),
                'CommentCount': statistics_data.get('commentCount'),
    
            }
            videos_data.append(video_data)


        # Check if we've collected enough data
        if len(videos_data) >= total_max_results:
            break

        # Check if there is a next page
        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break  # Exit the loop if there are no more results

    # Convert the list of dictionaries into a DataFrame
    videos_df = pd.DataFrame(videos_data)

    # Now, get channel statistics for each video
    channel_ids = list({item['snippet']['channelId'] for item in videos_response.get('items', [])})
    channels_request = youtube.channels().list(
        part='statistics',
        id=','.join(channel_ids)
    )
    channels_response = channels_request.execute()

    # Create a dictionary to hold channel statistics
    channel_stats = {}
    for item in channels_response.get('items', []):
        channel_id = item['id']
        channel_stats[channel_id] = item['statistics']

    # Add channel statistics to each video data
    for video in videos_data:
        channel_id = video['ChannelId']
        if channel_id in channel_stats:
            video['SubscriberCount'] = channel_stats[channel_id].get('subscriberCount', 'Not Available')
            video['TotalVideoCount'] = channel_stats[channel_id].get('videoCount', 'Not Available')
        else:
            video['SubscriberCount'] = 'Not Available'
            video['TotalVideoCount'] = 'Not Available'

    # Coalesce columns with 0 if none
    videos_df['ViewCount'] = videos_df['ViewCount'].fillna(0).astype(int)
    videos_df['LikeCount'] = videos_df['LikeCount'].fillna(0).astype(int)
    videos_df['DislikeCount'] = videos_df['DislikeCount'].fillna(0).astype(int)
    videos_df['CommentCount'] = videos_df['CommentCount'].fillna(0).astype(int)
    
    # Remove duplicates based on videos_df['VideoID']
    videos_df.drop_duplicates(subset='VideoID', keep='first', inplace=True)

    # Sort dataframe
    videos_df = videos_df.sort_values(by=['Duration', 'ViewCount'], ascending=[False, False])

    return videos_df

def duration_to_minutes(duration_str):
    # Parse the duration string into a timedelta object
    duration = isodate.parse_duration(duration_str)
    
    # Convert the duration to total seconds
    total_seconds = duration.total_seconds()
    
    # Convert seconds to minutes as a float
    minutes = total_seconds / 60
    
    return minutes

def import_to_airtable(df, base_id, table_id):
    table = api.table(base_id, table_id)
    print(table)

    # Read existing records from Airtable into a DataFrame
    all_records = table.all()
    existing_records = pd.DataFrame([record['fields'] for record in all_records])
    
    # Convert DataFrame to a list of dictionaries
    records = df.to_dict('records')

    # List to hold records that do not exist in Airtable
    new_records = []

    # Check each record if it exists in Airtable
    for record in records:
        video_id = record['VideoID']
        # Check if the VideoID already exists in the existing records
        if not existing_records[existing_records['VideoID'] == video_id].empty:
            continue  # Skip this record if it already exists
        new_records.append(record)
    
    # Use batch_create to import records
    # Airtable API limits batch operations to 10 records at a time
    for i in range(0, len(records), 10):
        batch = records[i:i + 10]
        table.batch_create(batch)

    return

def get_verified_interviews(base_id, table_id, person_name):
    table = api.table(base_id, table_id)
    all_records = table.all()
    df = pd.DataFrame([record['fields'] for record in all_records])

    # Filter the DataFrame based on the specified conditions
    filtered_df = df[(df['is_verified_interview'] == True) & (df['Query'] == person_name)]

    return filtered_df

def assign_classification(df, person_name):
    for index, row in df.iterrows():
        video_id = row['URL'].split('=')[-1]  # Extract the video ID from the YouTube URL
        
        # Use the existing youtube object
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        
        if 'items' in video_response and len(video_response['items']) > 0:
            snippet = video_response['items'][0]['snippet']
            video_title = snippet.get('title', '')
            video_description = snippet.get('description', '')
            
            # Check if the person in question is mentioned in the video title or description
            if person_name.lower() in video_title.lower() or person_name.lower() in video_description.lower():
                df.at[index, 'predicted_classification'] = 1
            else:
                df.at[index, 'predicted_classification'] = 0
        else:
            df.at[index, 'predicted_classification'] = 0
    
    return df

# This will only run if I run this as a standalone .py. 
# Aka this will NOT run if I run this file from app.py.
if __name__ == "__main__":
    # Set person name
    person_name = 'jeff bezos'
    # Get 200 long youtube videos
    df = search_videos_advanced(person_name, 200, 50, 'long')
    # Add probability column
    # df['probabilities'] = predict_proba(df, person_name)
    print(df[:10])

    print(assign_classification(df, 'Elon Musk')[:10])
    # Import into Airtable
    base_id = 'appFghYaLZCWgV7o5'
    table_id = 'tblrTXm6mQ6zagGdg'
    # tabletest = airtable_api.table(base_id, table_id)
    # print(tabletest)
    # import_to_airtable(df, base_id, table_id)
    
    # Export to csv
    # df.to_csv(f'{person_name}.csv', index=False)
    # # Export to json
    # json_result = df.to_json(orient='records', indent=4)
    # with open(f'{person_name}.json', 'w') as f:
    #     f.write(json_result)

# To dos
# - Repeat for other videos repos
# - Airtable -> Website 