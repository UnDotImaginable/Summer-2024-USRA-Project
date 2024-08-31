import requests
import csv
from datetime import datetime, timedelta
import os
import time

# API key and base URL (Try again without maxStations and maxDistance limiters)
api_key = '89XK53374PYPGTBEQGQMTLMEY' # Navid Key: 89XK53374PYPGTBEQGQMTLMEY, My key: FAQ9RUN3PPPZBHAPX4AKSM9PZ
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history"

# Ensure the directory exists
output_directory = 'Station_CSVData_ALL_STA'
os.makedirs(output_directory, exist_ok=True)

# Function to fetch data and append to CSV
def fetch_data(start_date, end_date, writer, include_header, location):
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
        'locations': location,
        #'maxStations': 1,
        #'maxDistance': 3000
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        content = response.content.decode('utf-8')
        lines = content.splitlines()
        reader = csv.reader(lines)
        
        if include_header:
            header = next(reader)
            header.extend(['Station Name', 'Station ID', 'Station Latitude', 'Station Longitude'])
            writer.writerow(header)
        else:
            next(reader)  # Skip the header line

        for row in reader:
            if any(row):
                if len(row) < 4:
                    row.extend([''] * (4 - len(row)))  # Fill missing station info
                station_info = row[-4:]
                row.extend(station_info)
                writer.writerow(row)
            
        print(f"Data from {start_date} to {end_date} for {location} has been successfully fetched.")
    else:
        print(f"Failed to retrieve data for {start_date} to {end_date} for {location}. HTTP Status code: {response.status_code}")
        print(f"Response content: {response.content.decode('utf-8')}")

# Loop through the date range and write to a single CSV file
def fetch_data_for_location(location, csv_filename):
    start_year = 2008
    end_year = 2023

    current_date = datetime(start_year, 1, 1)
    with open(f"{csv_filename}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        include_header = True
        
        while current_date.year < end_year:
            next_date = current_date + timedelta(days=61)
            if next_date.year > end_year or (next_date.year == end_year and next_date.month > 1):
                next_date = datetime(end_year, 1, 1)
            
            fetch_data(current_date, next_date, writer, include_header, location)
            include_header = False
            current_date = next_date + timedelta(days=1)
            time.sleep(1)  # Add delay to avoid rate limiting

    print(f"All data for {location} has been successfully fetched and exported to {csv_filename}.csv.")

# List of locations to fetch data for. 
locations = [
    #{"lat": 41.518, "lng": -81.684, "label": "CLE, BER, STR, BRU", "csv_name": "Cleveland Burke Lakefront Airport"}, # Cleveland Burke Lakefront Airport
    #{"lat": 41.52, "lng": -81.68, "label": "CLE, BER, STR, BRU", "csv_name": "KBKL"}, # KBKL
    #{"lat": 41.368, "lng": -81.702, "label": "CLE", "csv_name": "DW7394 Parma"}, # DW7394 Parma
    #{"lat": 41.406, "lng": -81.852, "label": "CLE, BER, STR, MED, BRU, LOD", "csv_name": "Hopkins Airport"}, # Hopkins Airport
    #{"lat": 41.42, "lng": -81.87, "label": "CLE, BER, STR, MED, BRU, LOD", "csv_name": "KCLE"}, # KCLE
    #{"lat": 41.68, "lng": -81.38, "label": "CLE", "csv_name": "KLNN"}, # KLNN
    #{"lat": 41.391, "lng": -81.795, "label": "BER", "csv_name": "DW8932 Brook Park"}, # DW8932 Brook Park
    #{"lat": 41.34, "lng": -82.18, "label": "BER, STR, MED, BRU, LOD, WOO, MAN", "csv_name": "KLPR"}, # KLPR
    #{"lat": 41.346, "lng": -82.179, "label": "BER, STR, MED, BRU, LOD, MAN", "csv_name": "Elyria Lorain Co. Airport"}, # Elyria Lorain Co. Airport
    #{"lat": 41.25, "lng": -81.813, "label": "STR", "csv_name": "EW2482 Brunswick"}, # EW2482 Brunswick
    #{"lat": 41.141, "lng": -81.853, "label": "MED, WOO", "csv_name": "FW2245 Medina"}, # FW2245 Medina
    #{"lat": 40.86, "lng": -81.88, "label": "MED, LOD, WOO, MAN", "csv_name": "KBJJ"}, # KBJJ
    #{"lat": 40.873, "lng": -81.887, "label": "MED, LOD, WOO, MAN", "csv_name": "Wooster Wayne Co. Airport"}, # Wooster Wayne Co. Airport
    #{"lat": 41.229, "lng": -81.816, "label": "BRU, LOD, WOO", "csv_name": "FW8255 Brunswick"}, # FW8255 Brunswick
    #{"lat": 41.04, "lng": -81.45, "label": "LOD, WOO", "csv_name": "KAKR"}, # KAKR
    #{"lat": 41.038, "lng": -81.464, "label": "LOD, WOO", "csv_name": "Akron Fulton International Airport"}, # Akron Fulton International Airport
    #{"lat": 40.82, "lng": -82.52, "label": "LOD, WOO, MAN, MV, GAM, FRED", "csv_name": "KMFD"}, # KMFD
    #{"lat": 40.82, "lng": -82.518, "label": "LOD, WOO, MAN, MV, GAM, FRED", "csv_name": "Mansfield Lahm Regional Airport"}, # Mansfield Lahm Regional Airport
    #{"lat": 40.764, "lng": -81.91, "label": "LOD, WOO, MAN", "csv_name": "Wooster 3 SSE"}, # Wooster 3 SSE
    #{"lat": 40.92, "lng": -81.43, "label": "WOO", "csv_name": "KCAK"}, # KCAK
    #{"lat": 40.918, "lng": -81.443, "label": "MED, WOO", "csv_name": "Akron Canton Airport"}, # Akron Canton Airport
    #{"lat": 40.47, "lng": -81.42, "label": "WOO", "csv_name": "KPHD"}, # KPHD
    #{"lat": 40.472, "lng": -81.424, "label": "WOO", "csv_name": "New Philadelphia Clever Field"}, # New Philadelphia Clever Field
    #{"lat": 40.77, "lng": -82.524, "label": "MAN, MV, FRED", "csv_name": "N8TWM-10 Mansfield"}, # N8TWM-10 Mansfield
    #{"lat": 40.333, "lng": -82.517, "label": "MAN, MV, GAM, FRED", "csv_name": "Knox Co. Airport"}, # Knox Co. Airport
    #{"lat": 40.33, "lng": -82.52, "label": "MAN, MV, GAM, FRED", "csv_name": "K4I3"}, # K4I3
    #{"lat": 40.61, "lng": -83.06, "label": "MAN", "csv_name": "KMNN"}, # KMNN
    #{"lat": 40.616, "lng": -83.064, "label": "MAN", "csv_name": "Marion Municipal Airport"}, # Marion Municipal Airport
    #{"lat": 40.29, "lng": -83.12, "label": "MAN, DEL, SUN, COL, DUB, WEST", "csv_name": "KDLZ"}, # KDLZ
    #{"lat": 40.073, "lng": -82.41, "label": "MV, GAM", "csv_name": "EW2896 Granville"}, # EW2896 Granville
    #{"lat": 40.023, "lng": -82.462, "label": "MV, GAM, FRED, PICK, LAN, CARR, PAT, ZANE", "csv_name": "Newark Heath Airport"}, # Newark Heath Airport
    #{"lat": 40.02, "lng": -82.46, "label": "MV, GAM, FRED, PICK, LAN, CARR, PAT", "csv_name": "KVTA"}, # KVTA
    #{"lat": 40.233, "lng": -81.923, "label": "GAM", "csv_name": "FW5361 Coshocton"}, # FW5361 Coshocton
    #{"lat": 40.28, "lng": -83.115, "label": "DEL, SUN, COL, DUB, WEST", "csv_name": "Delaware Municipal Jim Moore Field Airport"}, # Delaware Municipal Jim Moore Field Airport
    #{"lat": 40.204, "lng": -83.086, "label": "DEL, SUN", "csv_name": "CW4829 Powell"}, # CW4829 Powell
    #{"lat": 40.152, "lng": -83.27, "label": "DEL", "csv_name": "FW8882 Plain City"}, # FW8882 Plain City
    #{"lat": 40.078, "lng": -83.078, "label": "DEL, SUN, COL, DUB, WEST, REYN, CWIN", "csv_name": "Columbus Ohio State University Airport"}, # Columbus Ohio State University Airport
    #{"lat": 40.225, "lng": -83.352, "label": "DEL, DUB", "csv_name": "Union Co. Airport"}, # Union Co. Airport
    #{"lat": 40.22, "lng": -83.35, "label": "DEL, COL, DUB", "csv_name": "KMRT"}, # KMRT
    #{"lat": 40.08, "lng": -83.07, "label": "DEL, SUN, COL, DUB, WEST, REYN, CWIN", "csv_name": "KOSU"}, # KOSU
    #{"lat": 40.088, "lng": -82.802, "label": "SUN, WEST, REYN, PAT", "csv_name": "Marburn Academy WEATHERSTEM"}, # Marburn Academy WEATHERSTEM
    #{"lat": 40, "lng": -82.88, "label": "SUN, COL, WEST, REYN, CWIN, PICK, LAN, CARR, PAT", "csv_name": "KCMH"}, # KCMH
    #{"lat": 39.991, "lng": -82.887, "label": "SUN, COL, WEST, REYN, CWIN, PICK, LAN, CARR, PAT", "csv_name": "Port Columbus International Airport"}, # Port Columbus International Airport
    #{"lat": 40.002, "lng": -83.034, "label": "COL", "csv_name": "The Ohio State University WEATHERSTEM"}, # The Ohio State University WEATHERSTEM
    #{"lat": 40.055, "lng": -83.022, "label": "COL, WEST", "csv_name": "FW8432 Columbus"}, # FW8432 Columbus
    #{"lat": 39.9, "lng": -83.13, "label": "COL", "csv_name": "KTZR"}, # KTZR
    #{"lat": 39.9, "lng": -83.133, "label": "COL", "csv_name": "Columbus Bolton Field"}, # Columbus Bolton Field
    #{"lat": 39.76, "lng": -82.66, "label": "COL, REYN, CWIN, PICK, LAN, CARR, PAT, ZANE, CHIL", "csv_name": "KLHQ"}, # KLHQ
    #{"lat": 39.756, "lng": -82.657, "label": "COL, REYN, CWIN, PICK, LAN, CARR, PAT, ZANE, CHIL", "csv_name": "Lancaster Fairfield Co. Airport"}, # Lancaster Fairfield Co. Airport
    #{"lat": 39.93, "lng": -83.46, "label": "COL, JEFF", "csv_name": "KUYF"}, # KUYF
    #{"lat": 40.047, "lng": -83.081, "label": "DUB", "csv_name": "FW4387 Columbus"}, # FW4387 Columbus
    #{"lat": 40.015, "lng": -82.636, "label": "REYN, PICK, PAT", "csv_name": "CW7241 Pataskala"}, # CW7241 Pataskala
    #{"lat": 39.715, "lng": -82.61, "label": "CWIN, PICK, LAN, CARR", "csv_name": "CW5139 Lancaster"}, # CW5139 Lancaster
    #{"lat": 39.95, "lng": -81.9, "label": "ZANE", "csv_name": "KZZV"}, # KZZV
    #{"lat": 39.944, "lng": -81.892, "label": "ZANE", "csv_name": "Zanesville Municipal Airport"}, # Zanesville Municipal Airport
    #{"lat": 40.004, "lng": -82.081, "label": "ZANE", "csv_name": "Blue Rock"}, # Blue Rock
    #{"lat": 40.217, "lng": -81.882, "label": "ZANE", "csv_name": "KE8KMM Coshocton"}, # KE8KMM Coshocton
    #{"lat": 40.02, "lng": -82.46, "label": "ZANE", "csv_name": "KVTA"}, # KVTA
    #{"lat": 39.333, "lng": -83.017, "label": "CHIL", "csv_name": "EW9276 Chillicothe"}, # EW9276 Chillicothe
    #{"lat": 39.386, "lng": -82.985, "label": "CHIL", "csv_name": "Chillicothe"}, # Chillicothe
    #{"lat": 39.43, "lng": -83.01, "label": "CHIL, JEFF", "csv_name": "KRZT"}, # KRZT
    #{"lat": 39.981, "lng": -82.578, "label": "CHIL", "csv_name": "James A Rhodes"}, # James A Rhodes
    #{"lat": 39.217, "lng": -82.233, "label": "CHIL", "csv_name": "Ohio University Airport Snyder Field"}, # Ohio University Airport Snyder Field
    #{"lat": 39.21, "lng": -82.23, "label": "CHIL", "csv_name": "KUNI"}, # KUNI
    #{"lat": 39.095, "lng": -84.497, "label": "CIN", "csv_name": "EW5157 Newport"}, # EW5157 Newport
    #{"lat": 39.1, "lng": -84.42, "label": "CIN", "csv_name": "KLUK"}, # KLUK
    #{"lat": 39.103, "lng": -84.419, "label": "CIN", "csv_name": "Cincinnati Municipal Airport Lunken Field"}, # Cincinnati Municipal Airport Lunken Field
    #{"lat": 39.2, "lng": -84.607, "label": "CIN", "csv_name": "FW7513 Cincinnati"}, # FW7513 Cincinnati
    #{"lat": 39.044, "lng": -84.672, "label": "CIN", "csv_name": "Cincinnati Northern Kentucky International Airport"}, # Cincinnati Northern Kentucky International Airport
    #{"lat": 39.04, "lng": -84.67, "label": "CIN", "csv_name": "KCVG"}, # KCVG
    #{"lat": 39.08, "lng": -84.22, "label": "CIN", "csv_name": "KI69"}, # KI69
    #{"lat": 39.078, "lng": -84.21, "label": "CIN", "csv_name": "Clermont Co. Airport"}, # Clermont Co. Airport
    #{"lat": 39.393, "lng": -84.137, "label": "LEB", "csv_name": "EW6580 Lebanon"}, # EW6580 Lebanon
    #{"lat": 39.594, "lng": -84.226, "label": "LEB", "csv_name": "Dayton Wright Brothers Airport"}, # Dayton Wright Brothers Airport
    #{"lat": 39.6, "lng": -84.23, "label": "LEB", "csv_name": "KMGY"}, # KMGY
    #{"lat": 39.531, "lng": -84.395, "label": "LEB", "csv_name": "Middletown Hook Field Municipal Airport"}, # Middletown Hook Field Municipal Airport
    #{"lat": 39.36, "lng": -84.52, "label": "LEB", "csv_name": "KHAO"}, # KHAO
    #{"lat": 39.364, "lng": -84.525, "label": "LEB", "csv_name": "Hamilton Butler Co. Regional Airport"}, # Hamilton Butler Co. Regional Airport
    #{"lat": 39.43, "lng": -83.8, "label": "LEB, JEFF", "csv_name": "KILN"}, # KILN
    #{"lat": 39.426, "lng": -83.55, "label": "JEFF", "csv_name": "EW6534 Leesburg"}, # EW6534 Leesburg
    #{"lat": 39.716, "lng": -83.27, "label": "JEFF", "csv_name": "FW1217 Mount Sterling"}, # FW1217 Mount Sterling
    #{"lat": 39.431, "lng": -83.777, "label": "JEFF", "csv_name": "Wilmington Air Park"}, # Wilmington Air Park
    #{"lat": 39.83, "lng": -84.05, "label": "JEFF", "csv_name": "KFFO"}, # KFFO
    #{"lat": 39.833, "lng": -84.05, "label": "JEFF", "csv_name": "Dayton Wright Patterson AFB"}, # Dayton Wright Patterson AFB
    #{"lat": 40.86494, "lng": -82.31264, "label": "ASHLAND", "csv_name": "Ashland Weather Data"}, # Dayton Wright Patterson AFB
    #{"lat": 39.43332, "lng": -83.81671, "label": "WILMINGTON", "csv_name": "Wilmington Weather Data"}, # Dayton Wright Patterson AFB
]

# Fetch data for each location
for loc in locations:
    lat, lng = loc['lat'], loc['lng']
    csv_filename = os.path.join(output_directory, loc['csv_name'])
    fetch_data_for_location(f'{lat}, {lng}', csv_filename)

print("All data has been successfully fetched and exported to the respective CSV files.")
