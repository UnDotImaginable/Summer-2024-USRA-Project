import plotly.express as px
import dash
from dash import dcc, html, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
import csv
from datetime import timedelta
import numpy as np

def get_common_data_types(folders):
    print("Getting common data types from folders...")  # Debug statement
    common_data_types = set()
    for folder in folders:
        print(f"Checking folder: {folder}")  # Debug statement
        folder_data_types = set(os.listdir(folder))
        if not common_data_types:
            common_data_types = folder_data_types
        else:
            common_data_types.intersection_update(folder_data_types)
    print("Common data types:", common_data_types)  # Debug statement
    return sorted(common_data_types)

def create_dashboard(base_folder, folders):
    print("Initializing dashboard...")  # Debug statement

    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

    data_frequencies = ["daily.csv", 
                        "hourly.csv", 
                        "monthly.csv", 
                        "yearly.csv"]

    # Ensure the folders list is valid
    if not folders:
        print("No folders provided!")
        return

    # Prepend base_folder to each folder path
    folders = [os.path.join(base_folder, f) for f in folders]
    print("Folders provided:", folders)  # Debug statement

    # Get common data types
    data_types = get_common_data_types(folders)
    print("Common data types found:", data_types)  # Debug statement

    # Create dropdowns for data selection
    data_type_dropdown = dcc.Dropdown(
        id='select_data_type',
        options=[{'label': data_type, 'value': data_type} for data_type in data_types],
        value=data_types[0] if data_types else None,
        clearable=False
    )

    data_frequencies_dropdown = dcc.Dropdown(
        id='select_data_frequency',
        options=[{'label': i, 'value': i} for i in data_frequencies],
        value=data_frequencies[0],
        clearable=False
    )

    year_dropdown = dcc.Dropdown(
        id='select_year',
        clearable=False
    )

    month_dropdown = dcc.Dropdown(
        id='select_month',
        clearable=False
    )

    # Create the layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Regional Data Comparison"), width=6),
        ], justify='center'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph'), width=12),
        ]),
        dbc.Row([
            dbc.Col(data_type_dropdown, width=2),
            dbc.Col(data_frequencies_dropdown, width=2),
            dbc.Col(year_dropdown, width=2),
            dbc.Col(month_dropdown, width=2),
        ]),
    ], fluid=True)

    def load_and_prepare_data(file_path):
        print(f"Loading data from {file_path}")  # Debug statement
        df = pd.read_csv(file_path)

        if 'hourly.csv' in file_path:
            df = df.melt(var_name='Time', value_name='Value')
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        else:
            valid_columns = [col for col in df.columns if col.strip().replace("/", "").isdigit() or col in ['Time', 'Value']]
            df = df[valid_columns]
            df = df.melt(var_name='Time', value_name='Value')
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df = df.dropna(subset=['Time'])

        return df

    def get_data_type_options(folders):
        common_data_types = get_common_data_types(folders)
        return [{'label': data_type, 'value': data_type} for data_type in common_data_types]

    # Dropdown callback to update year options based on selected frequency
    @app.callback(
        Output('select_year', 'options'),
        [Input('select_data_frequency', 'value'),
         Input('select_data_type', 'value')]
    )
    def update_year_options(chosen_frequency, chosen_type):
        print(f"Updating year options for {chosen_type} and {chosen_frequency}")  # Debug statement
        if chosen_frequency and chosen_type:
            file_name = f"{folders[0]}/{chosen_type}/{chosen_frequency}"
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year
            years = df['Year'].unique()
            print("Available years:", years)  # Debug statement
            return [{'label': year, 'value': year} for year in years]
        return []

    # Dropdown callback to update month options based on selected year
    @app.callback(
        Output('select_month', 'options'),
        [Input('select_data_frequency', 'value'),
         Input('select_year', 'value'),
         Input('select_data_type', 'value')]
    )
    def update_month_options(chosen_frequency, chosen_year, chosen_type):
        print(f"Updating month options for {chosen_type}, {chosen_frequency}, and year {chosen_year}")  # Debug statement
        if chosen_frequency in ['daily.csv'] and chosen_year and chosen_type:
            file_name = f"{folders[0]}/{chosen_type}/{chosen_frequency}"
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year
            df['Month'] = df['Time'].dt.month
            df = df[df['Year'] == chosen_year]
            months = df['Month'].unique()
            print("Available months:", months)  # Debug statement
            return [{'label': month, 'value': month} for month in months]
        return []

    # Graph callback
    @app.callback(
        Output('graph', 'figure'),
        [Input('select_data_type', 'value'),
         Input('select_data_frequency', 'value'),
         Input('select_year', 'value'),
         Input('select_month', 'value')]
    )
    def update_graph(chosen_type, chosen_frequency, chosen_year, chosen_month):
        print(f"Updating graph for {chosen_type}, {chosen_frequency}, year {chosen_year}, month {chosen_month}")  # Debug statement
        base_path = "{}/{}/{}"
        combined_df = pd.DataFrame()  # Initialize an empty DataFrame to combine data
        avgs = []
        for folder in folders:
            file_name = base_path.format(folder, chosen_type, chosen_frequency)
            if not os.path.isfile(file_name):
                print(f"File not found: {file_name}")  # Debug statement
                continue
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year

            if chosen_frequency == 'yearly.csv':
                avgs.append([os.path.basename(folder), df['Value'].mean()])

            if chosen_year:
                df = df[df['Year'] == chosen_year]
            if chosen_month and 'Month' not in df.columns:
                df['Month'] = df['Time'].dt.month
            if chosen_month:
                df = df[df['Month'] == chosen_month]

            df['Region'] = os.path.basename(folder)  # Label the region by filename
            combined_df = pd.concat([combined_df, df])  # Combine data from all regions

        avgs = np.array(avgs)
        if chosen_frequency == 'yearly.csv':
            with open('yearZone.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for row in avgs:
                    writer.writerow(row)

        fig = px.scatter(combined_df, x='Time', y='Value', color='Region', 
                         title=f"{chosen_type} - {chosen_frequency.replace('.csv', '').capitalize()} Data Across Regions")

        if not combined_df.empty:
            if chosen_frequency == 'hourly.csv':
                if chosen_month and chosen_year:
                    min_date = combined_df['Time'].min()
                    fig.update_xaxes(range=[min_date, min_date + timedelta(days=1)])
                else:
                    min_date = combined_df['Time'].min()
                    max_date = combined_df['Time'].max()
                    fig.update_xaxes(range=[min_date, max_date])
            else:
                min_date = combined_df['Time'].min()
                max_date = combined_df['Time'].max()
                extra_days = pd.DateOffset(days=10)
                fig.update_xaxes(range=[min_date - extra_days, max_date + extra_days])

        return fig

    print("Starting Dash app...")  # Debug statement
    app.run_server(port=8053)
