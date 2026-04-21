import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


def plot(spot_accumulo, call_walls, put_walls, oi_totale, oi_calls_totale, oi_puts_totale, timestamps):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Spot con Walls', 'OI Totale'))

    x = [t.strftime("%H:%M:%S") for t in timestamps] if timestamps else list(range(1, len(spot_accumulo) + 1))

    fig.add_trace(go.Scatter(x=x, y=spot_accumulo, mode='lines', name='Spot',
                             line=dict(color='blue')), row=1, col=1)
    for val in call_walls["strike"].to_list():
        fig.add_hline(y=val, line=dict(color='red', dash='dash'), row=1, col=1)
    for val in put_walls["strike"].to_list():
        fig.add_hline(y=val, line=dict(color='green', dash='dash'), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=oi_totale, mode='lines', name='OI',
                             line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=oi_calls_totale, mode='lines', name='OI calls',
                             line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=oi_puts_totale, mode='lines', name='OI puts',
                             line=dict(color='red')), row=1, col=2)

    fig.update_layout(height=1000, width=1600)
    fig.write_html("main_analysis.html", auto_open=False)
    print("Grafico principale aggiornato in main_analysis.html")


def plot_cvd(cvd_history: list):
    if not cvd_history:
        return
    import polars as pl
    df = (
        pl.concat(cvd_history)
        .unique(subset=["open_time"], keep="last")
        .sort("open_time")
        .with_columns(pl.col("delta").cum_sum().alias("cvd"))
    )
    x = [datetime.fromtimestamp(t / 1000).strftime("%H:%M") for t in df["open_time"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["cvd"].to_list(), mode="lines", name="CVD",
                             line=dict(color="orange")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="CVD Spot (sessione completa)", xaxis_title="Ora",
                      yaxis_title="CVD", height=600, width=1000)
    fig.write_html("cvd_analysis.html", auto_open=False)
    print("Grafico CVD aggiornato in cvd_analysis.html")


def plot_iv_analysis(iv_skew_history, atm_iv_history, timestamps):
    if not iv_skew_history:
        return
    x = [t.strftime("%H:%M:%S") for t in timestamps] if timestamps else list(range(1, len(iv_skew_history) + 1))
    y_skew = [s if s is not None else float("nan") for s in iv_skew_history]
    y_atm = [v if v is not None else float("nan") for v in atm_iv_history]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("IV Skew 25d", "ATM IV"))

    fig.add_trace(go.Scatter(x=x, y=y_skew, mode="lines", name="IV Skew 25d",
                             line=dict(color="purple")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=y_atm, mode="lines", name="ATM IV",
                             line=dict(color="teal")), row=1, col=2)

    fig.update_yaxes(title_text="Skew (pp)", row=1, col=1)
    fig.update_yaxes(title_text="IV %", row=1, col=2)
    fig.update_layout(height=600, width=1400)
    fig.write_html("iv_analysis.html", auto_open=False)
    print("Grafico IV aggiornato in iv_analysis.html")


def plot_oi_profile(oi_history):
    if not oi_history:
        return
    top = oi_history[-1].head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["strike"].to_list(), y=top["oi_total"].to_list(), name="OI Total"))
    fig.update_layout(height=1000, width=1600)
    fig.write_html("second_analysis.html", auto_open=False)
    print("Grafico OI profile aggiornato in second_analysis.html")
