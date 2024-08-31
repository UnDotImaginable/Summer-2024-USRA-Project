# weather_data_fetcher.py
import requests
import csv
from datetime import datetime, timedelta

# API key and base URL
api_key = 'FAQ9RUN3PPPZBHAPX4AKSM9PZ'
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history"

# Function to fetch data and append to CSV
def fetch_data(start_date, end_date, location, writer, include_header):
    params = {
        'aggregateMinutes': '10',
        'combinationMethod': 'aggregate',
        'startDateTime': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'endDateTime': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'collectStationContributions': 'true',
        'contentType': 'csv',
        'unitGroup': 'metric',
        'locationMode': 'single',
        'key': api_key,
        'dataElements': 'All',
        'locations': location
    }
    
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        content = response.content.decode('utf-8')
        lines = content.splitlines()
        reader = csv.reader(lines)
        
        # Write header if needed
        if include_header:
            header = next(reader)
            header.extend(['Station Name', 'Station ID', 'Station Latitude', 'Station Longitude'])
            writer.writerow(header)
        
        for row in reader:
            station_info = row[-4:]  # Adjust this index based on actual position in response
            row.extend(station_info)
            writer.writerow(row)
            
        print(f"Data from {start_date} to {end_date} has been successfully fetched.")
    else:
        print(f"Failed to retrieve data for {start_date} to {end_date}. HTTP Status code: {response.status_code}")

# Function to write data to CSV for a given date range and location
def getWeatherData(output_filename, location, start_year, end_year):
    current_date = datetime(start_year, 1, 1)
    with open(output_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        include_header = True
        
        while current_date.year < end_year:
            next_date = current_date + timedelta(days=61)  # Approximately 2 months
            if next_date.year > end_year or (next_date.year == end_year and next_date.month > 1):
                next_date = datetime(end_year, 1, 1)
            
            fetch_data(current_date, next_date, location, writer, include_header)
            include_header = False  # Only include header for the first chunk
            current_date = next_date + timedelta(days=1)

    print("All data has been successfully fetched and exported to a single CSV file.")
    print(f"File saved as: {output_filename}")
