# governance/ai_governance_dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# Load compliance logs
df = pd.read_csv("logs/compliance_tracking.log")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI Governance & Compliance Dashboard"),
    dcc.Graph(
        figure=px.bar(df, x='policy', y='compliance_status', color='risk_level', title='Real-time Compliance Status')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)