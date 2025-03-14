# observability_dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# Load performance metrics
df = pd.read_csv("logs/performance_metrics.log")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("ML Model Observability Dashboard"),
    dcc.Graph(
        figure=px.line(df, x='timestamp', y=['latency', 'response_time'], title='Model Performance')
    ),
    dcc.Graph(
        figure=px.histogram(df, x='latency', title='Latency Distribution')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)