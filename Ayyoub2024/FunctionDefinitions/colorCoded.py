import plotly.express as px
import dash
from dash import dcc, html, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

def create_dashboard(data_path, data_types):
    print("Initializing dashboard...")  # Debug statement

    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

    data_frequencies = ["daily.csv", 
                        "hourly.csv", 
                        "monthly.csv", 
                        "yearly.csv"]

    data_type_dropdown = dcc.Dropdown(
        id='select_data_type',
        options=[{'label': i, 'value': i} for i in data_types],
        value=data_types[0],
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

    day_dropdown = dcc.Dropdown(
        id='select_day',
        clearable=False
    )

    area_slider = dcc.Slider(
        id='area_slider',
        min=0,
        max=150,
        step=10,
        value=50,
        marks={i: str(i) for i in range(0, 151, 10)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    )

    cp_slider = dcc.Slider(
        id='cp_slider',
        min=0,
        max=1,
        step=0.1,
        value=0.5,
        marks={i/10: str(i/10) for i in range(0, 11)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    )

    app.layout = dbc.Container([
        dcc.Store(id='slider-visibility', data={'visible': False}),

        dbc.Row([
            dbc.Col(html.H1("Data Visualization Dashboard"), width=6),
        ], justify='center'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph'), width=12),
        ]),
        dbc.Row([
            dbc.Col(data_type_dropdown, width=2),
            dbc.Col(data_frequencies_dropdown, width=2),
            dbc.Col(year_dropdown, width=2),
            dbc.Col(month_dropdown, width=2),
            dbc.Col(day_dropdown, width=2),
        ]),
        html.Div([
            dbc.Row([
                dbc.Col(html.Label('Area (A) in meters'), width=2),
                dbc.Col(dcc.Slider(
                    id='area_slider',
                    min=0,
                    max=150,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(0, 151, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=True
                ), width=6),
            ], id='area_slider_div', style={'display': 'none'}, justify='center'),
            dbc.Row([
                dbc.Col(html.Label('Coefficient of Performance (Cp)'), width=2),
                dbc.Col(dcc.Slider(
                    id='cp_slider',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(0, 11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=True
                ), width=6),
            ], id='cp_slider_div', style={'display': 'none'}, justify='center'),
        ])

    ], fluid=True)

    def load_and_prepare_data(file_path):
        print(f"Loading data from {file_path}")  # Debug statement
        df = pd.read_csv(file_path)
        
        if 'hourly.csv' in file_path:
            df = df.melt(var_name='Time', value_name='Value')
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        else:
            valid_columns = [col for col in df.columns if col.strip().replace("/", "").isdigit()]
            df = df[valid_columns]
            df = df.melt(var_name='Time', value_name='Value')
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df = df.dropna(subset=['Time'])
        return df

    def convertToWindPowerDensity(windSpeed, rho=1.225, A=1, Cp=0.5):
        windEnergy = 0.5 * rho * windSpeed**3 * A * Cp
        return windEnergy

    @app.callback(
        Output('select_year', 'options'),
        [Input('select_data_frequency', 'value')]
    )
    def update_year_options(chosen_frequency):
        print("Updating year options...")  # Debug statement
        if chosen_frequency in ['daily.csv', 'monthly.csv', 'hourly.csv']:
            file_name = f"{data_path}/{data_types[0]}/{chosen_frequency}"
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year
            years = df['Year'].unique()
            return [{'label': year, 'value': year} for year in years]
        return []

    @app.callback(
        Output('select_month', 'options'),
        [Input('select_data_frequency', 'value'),
         Input('select_year', 'value')]
    )
    def update_month_options(chosen_frequency, chosen_year):
        print("Updating month options...")  # Debug statement
        if chosen_frequency in ['daily.csv', 'hourly.csv'] and chosen_year:
            file_name = f"{data_path}/{data_types[0]}/{chosen_frequency}"
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year
            df['Month'] = df['Time'].dt.month
            df = df[df['Year'] == chosen_year]
            months = df['Month'].unique()
            return [{'label': month, 'value': month} for month in months]
        return []

    @app.callback(
        [Output('area_slider_div', 'style'),
         Output('cp_slider_div', 'style')],
        [Input('select_data_type', 'value')]
    )
    def toggle_sliders(selected_data_type):
        print(f"Toggling sliders for {selected_data_type}")  # Debug statement
        if selected_data_type == 'Wind Power Density':
            return {'display': 'block'}, {'display': 'block'}
        else:
            return {'display': 'none'}, {'display': 'none'}

    @app.callback(
        Output('select_day', 'options'),
        [Input('select_data_frequency', 'value'),
         Input('select_year', 'value'),
         Input('select_month', 'value')]
    )
    def update_day_options(chosen_frequency, chosen_year, chosen_month):
        print("Updating day options...")  # Debug statement
        if chosen_frequency == 'hourly.csv' and chosen_year and chosen_month:
            file_name = f"{data_path}/{data_types[0]}/{chosen_frequency}"
            df = load_and_prepare_data(file_name)
            df['Year'] = df['Time'].dt.year
            df['Month'] = df['Time'].dt.month
            df['Day'] = df['Time'].dt.day
            df = df[(df['Year'] == chosen_year) & (df['Month'] == chosen_month)]
            days = df['Day'].unique()
            return [{'label': day, 'value': day} for day in days]
        return []

    def apply_thresholds(df, thresholds):
        traces = []
        for threshold in thresholds:
            if threshold['Condition'] == 'Less than or equal to':
                mask = df['Value'] <= threshold['Value']
            else:
                mask = df['Value'] > threshold['Value']

            filtered_df = df[mask]
            traces.append(go.Scatter(
                x=filtered_df['Time'], 
                y=filtered_df['Value'], 
                mode='markers', 
                name=threshold['Label'],
                marker=dict(color=threshold['Color'])
            ))
            df = df[~mask]  # Remove rows that have already been assigned a color
        return traces

    @app.callback(
        Output('graph', 'figure'),
        [Input('select_data_type', 'value'),
         Input('select_data_frequency', 'value'),
         Input('select_year', 'value'),
         Input('select_month', 'value'),
         Input('select_day', 'value'),
         Input('area_slider', 'value'),
         Input('cp_slider', 'value')]
    )
    def update_graph(chosen_type, chosen_frequency, chosen_year, chosen_month, chosen_day, area, cp):
        print("Updating graph...")  # Debug statement
        file_name = f"{data_path}/{chosen_type}/{chosen_frequency}"
        if chosen_type == 'Wind Power Density':
            file_name = f"{data_path}/Wind Speed/{chosen_frequency}"
        df = load_and_prepare_data(file_name)
        df['Year'] = df['Time'].dt.year

        if chosen_frequency == 'daily.csv' and chosen_year:
            df = df[df['Year'] == chosen_year]
            if chosen_month:
                df['Month'] = df['Time'].dt.month
                df = df[df['Month'] == chosen_month]
            if chosen_day:
                df['Day'] = df['Time'].dt.day
                df = df[df['Day'] == chosen_day]

        elif chosen_frequency == 'hourly.csv' and chosen_year and chosen_month and chosen_day:
            df['Month'] = df['Time'].dt.month
            df['Day'] = df['Time'].dt.day
            df = df[(df['Year'] == chosen_year) & (df['Month'] == chosen_month) & (df['Day'] == chosen_day)]

        elif chosen_frequency == 'monthly.csv' and chosen_year:
            df = df[df['Year'] == chosen_year]

        if chosen_type == "Wind Power Density":
            df['Value'] = convertToWindPowerDensity(df['Value'], A=area, Cp=cp)

        fig = go.Figure()

        if chosen_type == "Wind Speed":
            thresholds = [
                {'Value': 2.7, 'Condition': 'Less than or equal to', 'Label': 'Sitting', 'Color': '#87CEFA'},
                {'Value': 3.8, 'Condition': 'Less than or equal to', 'Label': 'Standing', 'Color': '#3CB371'},
                {'Value': 4.7, 'Condition': 'Less than or equal to', 'Label': 'Strolling', 'Color': '#DAA520'},
                {'Value': 5.5, 'Condition': 'Less than or equal to', 'Label': 'Walking', 'Color': '#FF4500'},
                {'Value': 5.5, 'Condition': 'Greater than', 'Label': 'Uncomfortable', 'Color': '#B22222'},
                {'Value': 25, 'Condition': 'Greater than', 'Label': 'Exceeded', 'Color': '#8B0000'}
            ]
            with open('threshCount.txt', 'w') as f:
                for threshold in thresholds:
                    if threshold['Condition'] == 'Less than or equal to':
                        mask = df['Value'] <= threshold['Value']
                    else:
                        mask = df['Value'] > threshold['Value']
                    filtered_df = df[mask]
                    f.write(f"{threshold['Label']}: {len(filtered_df)}\n")
            traces = apply_thresholds(df, thresholds)
            for trace in traces:
                fig.add_trace(trace)
        else:
            fig = px.scatter(data_frame=df, x='Time', y='Value', title=f'{chosen_type} - {chosen_frequency[:-4].capitalize()} Resolution')

        if not df.empty:
            if chosen_frequency == 'hourly.csv':
                if chosen_day and chosen_month and chosen_year:
                    min_date = df['Time'].min()
                    fig.update_xaxes(range=[min_date, min_date + timedelta(days=1)])
                else:
                    min_date = df['Time'].min()
                    max_date = df['Time'].max()
                    fig.update_xaxes(range=[min_date, max_date])
            else:
                min_date = df['Time'].min()
                max_date = df['Time'].max()
                extra_days = pd.DateOffset(days=10)
                fig.update_xaxes(range=[min_date - extra_days, max_date + extra_days])

        fig.update_layout(title=f'{chosen_type} - {chosen_frequency[:-4].capitalize()} Resolution', showlegend=True)
        return fig

    print("Starting Dash app...")  # Debug statement
    app.run_server(port=8053)
