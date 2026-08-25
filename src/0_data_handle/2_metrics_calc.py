import json
import pandas as pd
import pandas_ta_classic as ta

feature_strategy = ta.Strategy(
    name="",
    ta=[
        # Momentum (20) ==================================================
        {"kind": "rsi"}, {"kind": "rsx"}, {"kind": "stoch"},
        {"kind": "stochrsi"}, {"kind": "macd"}, {"kind": "macdext"},
        {"kind": "roc"}, {"kind": "rocp"}, {"kind": "mom"},
        {"kind": "tsi"}, {"kind": "ppo"}, {"kind": "kst"},
        {"kind": "coppock"}, {"kind": "stochf"},

        # Trend (15) ==================================================
        {"kind": "adx"}, {"kind": "adxr"}, {"kind": "amat"},
        {"kind": "dx"}, {"kind": "vortex"}, {"kind": "vhf"},
        {"kind": "pmax"}, {"kind": "ttm_trend"}, {"kind": "qstick"},

        # Volatility (15) ==================================================
        {"kind": "atr"}, {"kind": "natr"}, {"kind": "bbands"},
        {"kind": "donchian"}, {"kind": "ui"}, {"kind": "hvol"},
        {"kind": "massi"}, {"kind": "thermo"}, {"kind": "true_range"},
        {"kind": "pdist"}, {"kind": "aberration"}, {"kind": "rvi"},
        {"kind": "kc"},

        # Volume (15) ==================================================
        {"kind": "obv"}, {"kind": "ad"}, {"kind": "adosc"},
        {"kind": "efi"}, {"kind": "pvt"}, {"kind": "wad"},

        # Moving Averages (15) ==================================================
        {"kind": "sma"}, {"kind": "ema"}, {"kind": "wma"},
        {"kind": "dema"}, {"kind": "tema"}, {"kind": "kama"},
        {"kind": "alma"}, {"kind": "t3"}, {"kind": "trima"},
        {"kind": "linreg"}, {"kind": "tsf"}, {"kind": "zlma"},
        {"kind": "hma"}, {"kind": "mama"},

        # Statistics (10) ==================================================
        {"kind": "stdev"}, {"kind": "variance"}, {"kind": "mad"},
        {"kind": "quantile"}, {"kind": "median"},

        # Performance (3)
        {"kind": "percent_return"}, {"kind": "log_return"}, {"kind": "drawdown"},
    ]
)

def process_tickers():
    tickers = []
    try:
        with open("_SETTINGS/top-10-full.json", "r") as f:
            tickers = json.load(f)
    except FileNotFoundError:
        print("Arquivo 'indexes.json' não encontrado.")
        return

    for ticker in tickers:
        df = pd.read_csv(f"_DATA/all/{ticker}.csv")
        df.ta.strategy(feature_strategy, cores=0)

        lines_cleaned = df.isnull().sum().sort_values(ascending=False).head(1).iloc[0]
        if lines_cleaned > 54:
            print(f"TICKER: {ticker} =========================")
            print(df.isnull().sum().sort_values(ascending=False).head(10))

        df.dropna(subset=df.columns.tolist(), inplace=True)
        df.to_csv(f"_DATA/featured/{ticker}.csv", index=False)



if __name__ == "__main__":
    process_tickers()