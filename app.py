from flask import Flask, render_template
import json
import os
from elon_musk_youtube_scraper import search_videos_advanced, predict_proba


app = Flask(__name__)

@app.route('/')
def index():
    # Specify the parameters directly here or configure them to be dynamic
    person_name = 'jeff bezos'
    max_results_per_request = 50
    total_max_results = 250
    video_length = 'long'
    
    # Call the function with parameters
    df = search_videos_advanced(person_name + ' interview', max_results_per_request, total_max_results, video_length)

    # Add probability column
    df['probabilities'] = predict_proba(df, person_name)
    print(df[:10])
    # Export to csv
    df.to_csv(f'{person_name}.csv', index=False)

    # Convert data to JSON for the template if necessary
    data_json = json.loads(df.to_json(orient='records'))
    
    return render_template('index.html', items=data_json
                           ,person_name=person_name)

if __name__ == '__main__':
    app.run(debug=True)