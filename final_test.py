import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from flask import Flask, request, jsonify
import json
import os
from Ayyoub2024.FunctionDefinitions.callAPIFunc import getWeatherData
from Ayyoub2024.FunctionDefinitions.dataOrganizationML import processCsvANN
import Ayyoub2024.FunctionDefinitions.ANN_WindSpeed as ann_windspeed
import Ayyoub2024.FunctionDefinitions.ANN_SolarRadiation as ann_solarradiation
from Ayyoub2024.FunctionDefinitions.callAPIFunc import getWeatherData
from Ayyoub2024.FunctionDefinitions.simulationOrganization import simulationProcess
import Ayyoub2024.FunctionDefinitions.colorCoded as color
import Ayyoub2024.FunctionDefinitions.regionComparison as region

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX])

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Iframe(id='map', srcDoc=open('assets/map.html', encoding='utf-8').read(), width='100%', height='600px', style={"border": "1px solid #ddd", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "borderRadius": "10px"})
                ])
            ], className="mb-4")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Selected Coordinates", className="card-title"),
                    html.Div(id='selected-coordinates', className="card-text"),
                    dbc.Button("Retrieve Weather Data", id='retrieve-data-button', color='primary', className="mt-2"),
                    dbc.Button("Predict Wind Speed", id='predict-wind-button', color='info', className="mt-2"),
                    dbc.Button("Predict Solar Radiation", id='predict-solar-button', color='warning', className="mt-2"),
                    html.Div(id='output-data', className="mt-3")
                ])
            ], className="mb-4")
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='weather-data-graph', style={"borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
        ])
    ])
], fluid=True, style={"backgroundColor": "#f8f9fa", "padding": "20px"})


# Endpoint to save coordinates
@server.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.json
    if 'lat' in data and 'lng' in data:
        with open('coordinates.json', 'a', encoding='utf-8') as f:
            json.dump(data, f)
            f.write('\n')
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

# Callback to display selected coordinates
@app.callback(
    Output('selected-coordinates', 'children'),
    Input('retrieve-data-button', 'n_clicks'),
    State('selected-coordinates', 'children')
)
def display_selected_coordinates(n_clicks, selected_coordinates):
    if n_clicks:
        if os.path.exists('coordinates.json'):
            with open('coordinates.json', 'r', encoding='utf-8') as f:
                data = f.readlines()
            if data:
                latest_coordinates = json.loads(data[-1])
                return f"Latitude: {latest_coordinates['lat']}, Longitude: {latest_coordinates['lng']}"
    return "No coordinates selected."

# Callback to retrieve weather data
@app.callback(
    Output('output-data', 'children'),
    Input('predict-wind-button', 'n_clicks'),
    Input('predict-solar-button', 'n_clicks'),
    State('selected-coordinates', 'children')
)
def process_weather_data_and_predict(predict_wind_n_clicks, predict_solar_n_clicks, selected_coordinates):
    ctx = dash.callback_context

    if not ctx.triggered or selected_coordinates == "No coordinates selected.":
        return "No data processed yet."

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    coords = selected_coordinates.split(',')
    lat = coords[0].split(':')[1].strip()
    lng = coords[1].split(':')[1].strip()

    location = f"{lat}, {lng}"
    output_filename = 'RetrievedWeatherData.csv'
    start_year = 2023
    end_year = 2024

    getWeatherData(output_filename, location, start_year, end_year)
    outputProcessed = 'ANNProcessed' + output_filename
    newData = processCsvANN(output_filename, outputProcessed)

    if button_id == 'predict-wind-button':
        model_save_path = 'wind_speed.pth'
        scaler_save_path = 'wind_speed_scaler.joblib'
        input_csv = outputProcessed
        output_csv = 'predictions_wind.csv'
        plot_path = 'C:/Users/veerg/Desktop/Renovatio (5 years)/predictions_wind.png'
        ann_windspeed.predict(model_save_path, scaler_save_path, input_csv, output_csv, plot_path)
        
        predictions_csv = 'predictions_wind.csv'
        best_week_plot_path = 'C:/Users/veerg/Desktop/Renovatio (5 years)/bestWeek_wind.png'
        ann_windspeed.find_best_week(predictions_csv, input_csv, best_week_plot_path)
        
        # Debug: print available columns to find the correct column name
        predictions = pd.read_csv(predictions_csv)
        print(predictions.columns)
        
        # Use the correct column name here after verifying it
        predictions_list = predictions['Predicted Wind Speed'].tolist()  # Adjust the column name if necessary
        
        predicted_values = [f"Day {i+1}: {value:.2f} m/s" for i, value in enumerate(predictions_list[:30])]
        
        return html.Div([
            html.H5("Wind Speed Predictions Generated Successfully"),
            html.Img(src=plot_path),
            html.Img(src=best_week_plot_path),
            html.H6("Predicted Wind Speeds for the Next 30 Days:"),
            html.Ul([html.Li(pred) for pred in predicted_values])
        ])

    elif button_id == 'predict-solar-button':
        model_save_path = 'solar_radiation.pth'
        scaler_save_path = 'solar_radiation_scaler.joblib'
        input_csv = outputProcessed
        output_csv = 'predictions_solar.csv'
        plot_path = 'C:/Users/veerg/Desktop/Renovatio (5 years)/predictions_solar.png'
        ann_solarradiation.predict(model_save_path, scaler_save_path, input_csv, output_csv, plot_path)
        
        predictions_csv = 'predictions_solar.csv'
        best_week_plot_path = 'C:/Users/veerg/Desktop/Renovatio (5 years)/bestWeek_solar.png'
        ann_solarradiation.find_best_week(predictions_csv, input_csv, best_week_plot_path)
        
        # Use the correct column name for solar radiation predictions
        predictions_list = predictions['Predicted Solar Radiation'].tolist()  # Adjust the column name if necessary
        
        predicted_values = [f"Day {i+1}: {value:.2f} W/m²" for i, value in enumerate(predictions_list[:30])]
        
        return html.Div([
            html.H5("Solar Radiation Predictions Generated Successfully"),
            html.Img(src=plot_path),
            html.Img(src=best_week_plot_path),
            html.H6("Predicted Solar Radiation for the Next 30 Days:"),
            html.Ul([html.Li(pred) for pred in predicted_values])
        ])

    return "No data processed yet."


if __name__ == '__main__':
    app.run_server(debug=True)
