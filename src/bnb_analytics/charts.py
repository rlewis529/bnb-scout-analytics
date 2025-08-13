import io
import matplotlib.pyplot as plt
import pandas as pd

def chart_avg_price_by_neighborhood(df: pd.DataFrame):
    avg = (
        df.groupby("neighbourhood_cleansed")["price"]
          .mean()
          .sort_values(ascending=False)
    )
    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    avg.plot(kind="bar", edgecolor="black")
    ax.set_title("Average Nightly Price by Neighborhood")
    ax.set_ylabel("Average Price ($)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def chart_feature_importance(fi: pd.DataFrame, k: int = 12):
    top = fi.head(k).sort_values(by="coefficient")
    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.barh(top["feature"], top["coefficient"], edgecolor="black")
    ax.axvline(0)
    ax.set_title("Top Features by Impact on Price")
    ax.set_xlabel("Price Impact ($)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
