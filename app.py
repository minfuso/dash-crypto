import os
import json
import time
import requests
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# --- Fonction de cache ---
def get_data_with_cache(cache_file, url, params=None, max_age=1800, key_check=None):
    """
    Récupère les données depuis un cache fichier ou l'API si expiré/incomplet.

    cache_file : chemin du fichier cache
    url        : endpoint API
    params     : paramètres GET
    max_age    : durée de validité en secondes (1800 = 30 min)
    key_check  : clé à vérifier dans les données (ex: 'lastPrice')
    """
    if os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < max_age:
            with open(cache_file, "r") as f:
                try:
                    data = json.load(f)
                    if key_check is None or key_check in data:
                        return data
                except Exception:
                    pass

    response = requests.get(url, params=params)
    data = response.json()

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data


# --- Application Dash ---
app = dash.Dash(__name__)
symbols = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "cardano": "ADAUSDT",
    "solana": "SOLUSDT"
}

app.layout = html.Div([
    html.H1("📊 Dashboard Crypto (Binance API)", style={"textAlign": "center"}),
    
    <div>
    
    </div>

    dcc.Dropdown(
        id="crypto-dropdown",
        options=[{"label": c.capitalize(), "value": c} for c in symbols.keys()],
        value="bitcoin",
        style={"width": "50%"}
    ),

    html.Div(id="crypto-metrics", style={"margin": "20px 0", "fontSize": "20px"}),

    dcc.Graph(id="crypto-graph")
])


@app.callback(
    Output("crypto-metrics", "children"),
    Input("crypto-dropdown", "value")
)
def update_metrics(crypto):
    symbol = symbols[crypto]
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = get_data_with_cache(f"cache_{symbol}_price.json", url,
                               params={"symbol": symbol}, max_age=1800, key_check="lastPrice")

    price = float(data["lastPrice"])
    change = float(data["priceChangePercent"])

    return [
        html.H2(f"{crypto.capitalize()}"),
        html.P(f"Prix actuel : {price:.2f} $"),
        html.P(f"Variation 24h : {change:.2f} %")
    ]


@app.callback(
    Output("crypto-graph", "figure"),
    Input("crypto-dropdown", "value")
)
def update_graph(crypto):
    symbol = symbols[crypto]
    url = "https://api.binance.com/api/v3/klines"
    data = get_data_with_cache(f"cache_{symbol}_history.json", url,
                               params={"symbol": symbol, "interval": "1h", "limit": 168},
                               max_age=1800)

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "nb_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)

    fig = px.line(df, x="timestamp", y="close",
                  title=f"Évolution sur 7 jours ({crypto.capitalize()})")
    
    fig.update_layout(
        plot_bgcolor="#1f2039",   # fond du graphe
        paper_bgcolor="#1f2039",  # fond autour du graphe
        font=dict(color="#f9f8ff"),  # couleur du texte
        title=dict(font=dict(size=20, color="#f9f8ff"))  # style du titre
    )

    fig.update_traces(
        line=dict(color="#a5b4fc", width=3),  # couleur + épaisseur du tracé
        marker=dict(size=4)                   # taille des points si visibles
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
