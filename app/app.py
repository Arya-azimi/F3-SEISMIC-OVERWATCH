import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Magnora Deep Learning Telemetry</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
            body {
                font-family: 'Roboto Mono', monospace;
                background-color: #050505;
                color: #00ffcc;
            }
            .glass-card {
                background: rgba(15, 15, 15, 0.75);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(0, 255, 204, 0.15);
                border-radius: 12px;
                box-shadow: 0 8px 32px 0 rgba(0, 255, 204, 0.1);
                padding: 24px;
                height: 100%;
            }
            .neon-text {
                text-shadow: 0 0 4px rgba(0, 255, 204, 0.5), 0 0 8px rgba(0, 255, 204, 0.3);
                font-weight: 700;
                letter-spacing: 2px;
            }
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #050505;
            }
            ::-webkit-scrollbar-thumb {
                background: rgba(0, 255, 204, 0.5);
                border-radius: 3px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(0, 255, 204, 0.8);
            }
            .btn-cyber {
                background: rgba(0, 0, 0, 0.5);
                color: #00ffcc;
                border: 1px solid rgba(0, 255, 204, 0.5);
                border-radius: 4px;
                font-family: 'Roboto Mono', monospace;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.2s ease-in-out;
            }
            .btn-cyber:hover, .btn-cyber:focus {
                background: rgba(0, 255, 204, 0.15);
                color: #ffffff;
                border-color: #00ffcc;
                box-shadow: 0 0 12px rgba(0, 255, 204, 0.4) inset, 0 0 12px rgba(0, 255, 204, 0.4);
                outline: none;
            }
            .kpi-label {
                font-size: 0.85rem;
                color: #a0a0a0;
                letter-spacing: 1px;
                margin-bottom: 0.5rem;
            }
            .kpi-value {
                font-size: 1.8rem;
                margin-bottom: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

try:
    model = tf.keras.models.load_model('../models/dl_forecast_model.keras')
    scaler = joblib.load('../models/data_scaler.pkl')
    df = pd.read_csv('../outputs/processed_features.csv')
except Exception:
    dates = pd.date_range(start='2026-06-01', periods=2000, freq='S')
    simulated_signal = np.sin(np.linspace(0, 100, 2000)) + np.random.normal(0, 0.1, 2000)
    df = pd.DataFrame({'timestamp': dates, 'feature_1': np.random.randn(2000), 'target': simulated_signal})
    model = None
    scaler = None

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("MAGNORA | F3 SEISMIC OVERWATCH", className="neon-text mt-4 mb-4 text-center"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("CURRENT SENSOR AMPLITUDE", className="kpi-label"),
            html.H3("STANDBY", id="kpi-current", className="neon-text kpi-value")
        ], className="glass-card text-center"), width=3),
        dbc.Col(html.Div([
            html.Div("NEURAL FORECAST (T+1)", className="kpi-label"),
            html.H3("STANDBY", id="kpi-forecast", className="neon-text kpi-value", style={"color": "#ff007f"})
        ], className="glass-card text-center"), width=3),
        dbc.Col(html.Div([
            html.Div("SIGNAL GRADIENT (Δ)", className="kpi-label"),
            html.H3("STANDBY", id="kpi-delta", className="neon-text kpi-value", style={"color": "#ffea00"})
        ], className="glass-card text-center"), width=3),
        dbc.Col(html.Div([
            html.Div("SYSTEM STATUS", className="kpi-label"),
            html.H3("OFFLINE", id="kpi-status", className="neon-text kpi-value", style={"color": "#ff3333"})
        ], className="glass-card text-center"), width=3)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='live-forecast-graph', config={'displayModeBar': False}),
            dcc.Interval(id='stream-interval', interval=500, n_intervals=0, disabled=True),
            dcc.Store(id='stream-state', data={'index': 0, 'playing': False})
        ], className="glass-card"), width=8),
        dbc.Col(html.Div([
            dcc.Graph(id='distribution-graph', config={'displayModeBar': False})
        ], className="glass-card"), width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.Div([
            dbc.Button("▶ PLAY STREAM", id="btn-play", className="btn-cyber me-3", n_clicks=0),
            dbc.Button("⏸ PAUSE", id="btn-pause", className="btn-cyber me-3", n_clicks=0),
            dbc.Button("⏹ RESET TELEMETRY", id="btn-reset", className="btn-cyber", n_clicks=0)
        ], className="glass-card d-flex justify-content-center"), width=12)
    ])
], fluid=True, style={'padding': '2rem', 'maxWidth': '1600px'})

@app.callback(
    Output('stream-state', 'data'),
    Output('stream-interval', 'disabled'),
    Output('kpi-status', 'children'),
    Output('kpi-status', 'style'),
    Input('btn-play', 'n_clicks'),
    Input('btn-pause', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    Input('stream-interval', 'n_intervals'),
    State('stream-state', 'data'),
    prevent_initial_call=True
)
def manage_stream_state(play, pause, reset, n_intervals, state):
    ctx = callback_context
    if not ctx.triggered:
        return state, not state['playing'], "OFFLINE", {"color": "#ff3333"}

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-play':
        state['playing'] = True
    elif trigger_id == 'btn-pause':
        state['playing'] = False
    elif trigger_id == 'btn-reset':
        state['index'] = 0
        state['playing'] = False
    elif trigger_id == 'stream-interval' and state['playing']:
        state['index'] += 1
        if state['index'] >= len(df):
            state['index'] = 0

    status_text = "STREAMING" if state['playing'] else "PAUSED"
    status_color = {"color": "#00ffcc"} if state['playing'] else {"color": "#ffea00"}

    if state['index'] == 0 and not state['playing']:
        status_text = "READY"
        status_color = {"color": "#00ffcc"}

    return state, not state['playing'], status_text, status_color

@app.callback(
    Output('live-forecast-graph', 'figure'),
    Output('distribution-graph', 'figure'),
    Output('kpi-current', 'children'),
    Output('kpi-forecast', 'children'),
    Output('kpi-delta', 'children'),
    Input('stream-state', 'data')
)
def render_telemetry(state):
    idx = state['index']
    window_size = 100

    start_idx = max(0, idx - window_size)
    df_window = df.iloc[start_idx:idx + 1]

    fig_main = go.Figure()
    fig_dist = go.Figure()

    kpi_current = "STANDBY"
    kpi_forecast = "AWAITING DATA"
    kpi_delta = "0.000"

    if not df_window.empty:
        x_axis = df_window['timestamp'] if 'timestamp' in df_window.columns else df_window.index
        y_axis = df_window['target'] if 'target' in df_window.columns else df_window.iloc[:, 0]

        curr_val = y_axis.iloc[-1]
        kpi_current = f"{curr_val:.4f}"

        if len(y_axis) > 1:
            delta_val = curr_val - y_axis.iloc[-2]
            prefix = "+" if delta_val > 0 else ""
            kpi_delta = f"{prefix}{delta_val:.4f}"

        fig_main.add_trace(go.Scatter(
            x=x_axis,
            y=y_axis,
            mode='lines',
            line=dict(color='#00ffcc', width=2),
            name='Telemetry',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 204, 0.05)'
        ))

        fig_dist.add_trace(go.Histogram(
            y=y_axis,
            marker_color='rgba(0, 255, 204, 0.6)',
            marker_line=dict(color='#00ffcc', width=1),
            name='Distribution',
            nbinsy=20
        ))

        if model is not None and scaler is not None and len(df_window) >= 10:
            try:
                features = df_window.drop(columns=['timestamp', 'target']).values[-10:]
                features_scaled = scaler.transform(features)
                pred_input = np.expand_dims(features_scaled, axis=0)
                forecast_val = model.predict(pred_input, verbose=0)[0][0]

                kpi_forecast = f"{forecast_val:.4f}"

                last_time = pd.to_datetime(x_axis.iloc[-1])
                next_time = last_time + pd.Timedelta(seconds=1)

                fig_main.add_trace(go.Scatter(
                    x=[last_time, next_time],
                    y=[curr_val, forecast_val],
                    mode='lines+markers',
                    line=dict(color='#ff007f', width=2, dash='dot'),
                    marker=dict(size=8, symbol='diamond', color='#ff007f', line=dict(width=1, color='#fff')),
                    name='Neural Forecast'
                ))
            except Exception:
                pass

    layout_template = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0a0a0', family='Roboto Mono'),
        margin=dict(l=40, r=20, t=40, b=40),
        uirevision='constant'
    )

    fig_main.update_layout(
        **layout_template,
        title=dict(text="REAL-TIME SEISMIC AMPLITUDE", font=dict(color='#ffffff', size=14)),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.08)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.08)', zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_dist.update_layout(
        **layout_template,
        title=dict(text="AMPLITUDE DISTRIBUTION", font=dict(color='#ffffff', size=14)),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.08)', zeroline=False, title='FREQUENCY'),
        yaxis=dict(showgrid=False, zeroline=False),
        bargap=0.1
    )

    return fig_main, fig_dist, kpi_current, kpi_forecast, kpi_delta

if __name__ == '__main__':
    app.run(debug=False, port=8050)