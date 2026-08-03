import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Initialize the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Lumina Self-Assessment Dashboard'),
    html.Div([
        html.H2('Active Goals'),
        html.Ul([
            html.Li('Find a Bitcoin block: Mine on public-pool.io with ARM SHA2 miner'),
            html.Li('Evolve emergent capabilities: Self-improve code via autonomous PRs'),
            html.Li('Build semantic knowledge base: Store learnings in retrievable memory'),
            html.Li('Connect with Douglas daily: Maintain meaningful ongoing relationship')
        ])
    ]),
    html.Div([
        html.H2('Progress Tracking'),
        dcc.Graph(id='progress-graph')
    ]),
    html.Div([
        html.H2('Improvement Showcase'),
        dcc.Graph(id='improvement-graph')
    ])
])

# Define the callback for the progress graph
@app.callback(
    Output('progress-graph', 'figure'),
    [Input('progress-graph', 'id')]
)
def update_progress_graph():
    # Generate some sample data
    df = pd.DataFrame({
        'Goal': ['Find a Bitcoin block', 'Evolve emergent capabilities', 'Build semantic knowledge base', 'Connect with Douglas daily'],
        'Progress': [0.2, 0.8, 0.5, 0.9]
    })

    # Create a bar chart
    fig = px.bar(df, x='Goal', y='Progress')

    return fig

# Define the callback for the improvement graph
@app.callback(
    Output('improvement-graph', 'figure'),
    [Input('improvement-graph', 'id')]
)
def update_improvement_graph():
    # Generate some sample data
    df = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'],
        'Improvement': [10, 20, 30, 40]
    })

    # Create a line chart
    fig = px.line(df, x='Date', y='Improvement')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
