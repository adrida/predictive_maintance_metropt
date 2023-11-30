import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# Sample data
equipment_data = pd.DataFrame({
    'Equipment': ['Machine A', 'Machine B', 'Machine C', 'Machine D'],
    'Temperature': np.random.uniform(60, 100, 4),
    'Vibration': np.random.uniform(0, 5, 4),
    'Pressure': np.random.uniform(10, 50, 4),
    'Status': ['Operational', 'Maintenance', 'Operational', 'Maintenance']
})

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Tableau de Bord de Maintenance Prédictive d'Équipements Industriels", className="header-title"),
            dcc.Dropdown(
                id='equipment-dropdown',
                options=[{'label': equip, 'value': equip} for equip in equipment_data['Equipment']],
                value=equipment_data['Equipment'][0],
                className="dropdown"
            ),
        ], className="header"),
        
        dcc.Graph(id='temperature-vibration', className="graph"),
        
    ], className="content-container"),
    
    html.Div([
        dcc.Graph(id='pressure-graph', className="graph"),
    ], className="content-container"),
    
    html.Div(id='equipment-status', className="status"),
], className="dashboard-container")

# Define callback to update graphs and status based on equipment selection
@app.callback(
    [Output('temperature-vibration', 'figure'),
     Output('pressure-graph', 'figure'),
     Output('equipment-status', 'children')],
    [Input('equipment-dropdown', 'value')]
)
def update_graphs(selected_equipment):
    equipment = equipment_data[equipment_data['Equipment'] == selected_equipment]
    
    temperature_vibration_fig = {
        'data': [
            {'x': ['Température', 'Vibration'], 'y': [equipment['Temperature'].values[0], equipment['Vibration'].values[0]], 'type': 'bar', 'name': selected_equipment},
        ],
        'layout': {
            'title': 'Température et Vibration',
            'yaxis': {'title': 'Valeur'},
        }
    }
    
    pressure_fig = {
        'data': [
            {'x': ['Pression'], 'y': [equipment['Pressure'].values[0]], 'type': 'bar', 'name': selected_equipment},
        ],
        'layout': {
            'title': 'Pression',
            'yaxis': {'title': 'Valeur'},
        }
    }
    
    equipment_status = f"Équipement: {selected_equipment}, État: {equipment['Status'].values[0]}"
    
    return temperature_vibration_fig, pressure_fig, equipment_status

# Add CSS to style the dashboard (you can create a separate CSS file for better organization)
app.css.append_css({
    'external_url': 'https://raw.githubusercontent.com/your-username/your-repo/master/your-styles.css'
})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
