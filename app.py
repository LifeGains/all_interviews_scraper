from flask import Flask, render_template
import json
import os
from elon_musk_youtube_scraper import search_videos_advanced, assign_classification, import_to_airtable, get_verified_interviews

app = Flask(__name__)

@app.route('/')
def index():
    # Specify the parameters directly here or configure them to be dynamic
    person_name = 'tobi lutke'
    max_results_per_request = 50
    total_max_results = 250
    video_length = 'long'
    
    # Call the function with parameters
    df = search_videos_advanced(person_name + ' interview', max_results_per_request, total_max_results, video_length)

    # Add classification column
    df = assign_classification(df, person_name)

    # Add querytag column
    df['Query'] = person_name

    print(df[:10])

    # Import into Airtable
    base_id = 'appFghYaLZCWgV7o5'
    table_id = 'tblrTXm6mQ6zagGdg'
    # tabletest = airtable_api.table(base_id, table_id)
    # print(tabletest)
    import_to_airtable(df, base_id, table_id)

    # Export all results to csv
    df.to_csv(f'{person_name}.csv', index=False)

    # Get verified interviews from Airtable
    verified_interviews_df = get_verified_interviews(base_id, table_id, person_name)

    # Convert data to JSON for Flask to render.
    data_json = json.loads(verified_interviews_df.to_json(orient='records'))
    
    return render_template('index.html', items=data_json
                           ,person_name=person_name)

if __name__ == '__main__':
    app.run(debug=True)