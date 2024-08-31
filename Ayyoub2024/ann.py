from FunctionDefinitions.callAPIFunc import getWeatherData
from FunctionDefinitions.dataOrganizationML import processCsvANN
import FunctionDefinitions.ANN_WindSpeed as ann_windspeed
import FunctionDefinitions.ANN_SolarRadiation as ann_solarradiation
import os

# State your intentions
GatherDataFromWebAPI = False  # Step 1 for both ANN and Simulation
ProcessDataForANNTraining = True  # Step 2 for ANN
TrainANN = True  # Step 3 for ANN
PredictANN = True  # Step 4 for ANN
FindBestWeekANN = True  # Step 4 for ANN (optional)

Name = 'DW7394 Parma.csv'
Name_without_extension = os.path.splitext(Name)[0]

# Calling API Function, input params to change
output_filename = Name_without_extension + '.csv'
location = '41.368, -81.702'  # Example location (latitude, longitude) CAN ALSO BE IN ADDRESS FORMAT
start_year = 2015
end_year = 2021
if GatherDataFromWebAPI:
    getWeatherData(output_filename, location, start_year, end_year)

# Calling ANN processing function
readFileName = output_filename
outputProcessed = 'ANNProcessed_' + output_filename
if ProcessDataForANNTraining:
    newData = processCsvANN(readFileName, outputProcessed)

input_csv = outputProcessed

# Dictionary to map target names to their corresponding modules
ann_modules = {
    'Wind Speed': ann_windspeed,
    'Solar Radiation': ann_solarradiation
}

targets = ['Wind Speed', 'Solar Radiation']

for target in targets:
    model_path = f'{target.replace(" ", "_").lower()}.pth'
    scaler_path = f'{target.replace(" ", "_").lower()}_scaler.joblib'
    output_csv = f'predictions_{target.replace(" ", "_").lower()}.csv'
    plot_path = f'predictions_{target.replace(" ", "_").lower()}.png'
    best_week_plot_path = f'bestWeek_{target.replace(" ", "_").lower()}.png'

    # Train the ANN model
    if TrainANN:
        ann_modules[target].train_and_evaluate_lstm(input_csv, model_path, scaler_path)

    # Predict using the trained ANN model
    if PredictANN:
        getWeatherData(Name_without_extension + '_2023.csv', location, 2023, 2024)
        new_data = processCsvANN(Name_without_extension + '_2023.csv', 'ANNProcessed_' + Name_without_extension + '_2023.csv')
        input_csv = 'ANNProcessed_' + Name_without_extension + '_2023.csv'
        ann_modules[target].predict(model_path, scaler_path, input_csv, output_csv, plot_path)

    # Find the best week
    if FindBestWeekANN:
        ann_modules[target].find_best_week(output_csv, input_csv, best_week_plot_path)
