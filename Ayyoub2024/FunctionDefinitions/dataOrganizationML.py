import pandas as pd
import traceback

def processCsvANN(file_path: str, output_file_path: str) -> pd.DataFrame:
    try:
        # Load the dataset from the CSV file
        data = pd.read_csv(file_path, low_memory=False)

        # Convert 'Date time' column to datetime format, handling errors
        data['Date time'] = pd.to_datetime(data['Date time'], errors='coerce')

        # Drop rows with invalid 'Date time' values
        data = data.dropna(subset=['Date time'])

        # Convert 'Wind Speed' column to numeric, handling errors
        if 'Wind Speed' in data.columns:
            data['Wind Speed'] = pd.to_numeric(data['Wind Speed'], errors='coerce')
            data['Wind Speed'] = data['Wind Speed'] / 3.6

        # Convert other relevant columns to numeric
        numeric_columns = [
            'Temperature', 'Dew Point', 'Wind Direction', 'Precipitation',
            'Wind Gust', 'Solar Radiation', 'Relative Humidity', 
            'Cloud Cover', 'Sea Level Pressure'
        ]
        for column in numeric_columns:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')

        # Create 'day_of_year' and 'time' columns
        data['day_of_year'] = data['Date time'].dt.strftime('%m/%d')
        data['day_of_year'] = pd.to_datetime(data['day_of_year'], format='%m/%d', errors='coerce')
        data['day_of_year'] = (data['day_of_year'].dt.dayofyear - 1).fillna(60)
        data['time'] = data['Date time'].dt.strftime('%H:%M:%S')
        time = pd.to_datetime(data['time'], format='%H:%M:%S', errors='coerce')
        data['time'] = (time.dt.hour * 6 + time.dt.minute // 10).fillna(0)

        # Add year column
        data['year'] = data['Date time'].dt.strftime('%Y')
        data['year'] = pd.to_numeric(data['year'], errors='coerce')

        # Include only the relevant columns
        columns_to_include = [
            'year',
            'day_of_year', 'time', 'Temperature', 'Dew Point', 'Wind Direction', 
            'Precipitation', 'Wind Gust', 'Solar Radiation', 'Relative Humidity', 
            'Cloud Cover', 'Sea Level Pressure', 'Wind Speed'
        ]
        columns_to_include = [col for col in columns_to_include if col in data.columns]

        # Create a new dataframe with the selected columns
        processed_data = data[columns_to_include]

        # Handle missing values for 'Solar Radiation'
        if 'Solar Radiation' in processed_data.columns:
            # For missing values during the day, use interpolation
            processed_data['Solar Radiation'] = processed_data['Solar Radiation'].interpolate(method='linear', limit_direction='both')
            
            # Ensure that nighttime hours (e.g., from 8 PM to 6 AM) have a value of 0
            nighttime_hours = (processed_data['time'] >= 18 * 6) | (processed_data['time'] <= 6 * 6)
            processed_data.loc[nighttime_hours, 'Solar Radiation'] = 0

        # Drop rows where 'time' is NaN
        processed_data = processed_data.dropna(subset=['time'])

        # Fill NaN values for other columns with their respective column averages
        for column in columns_to_include:
            if column not in ['Solar Radiation', 'time']:
                if processed_data[column].dtype in ['float64', 'int64']:
                    # If the column is empty, fill it with zeroes
                    if processed_data[column].isnull().all():
                        processed_data[column] = 0
                    else:
                        processed_data[column] = processed_data[column].interpolate(method='linear', limit_direction='both')

        # Save the processed data to a new CSV file
        if output_file_path != '':
            processed_data.to_csv(output_file_path, index=False)

        return processed_data

    except Exception as e:
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Error processing file {file_path}:\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n\n")
        return pd.DataFrame()
