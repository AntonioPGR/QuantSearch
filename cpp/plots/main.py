import pandas as pd
import matplotlib.pyplot as plt

RUN_ID="20261020_221717_605"

df = pd.read_csv(f"../_results/{RUN_ID}.csv")
df["date"] = pd.to_datetime(df["date"])

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(df["date"], df["my_money"], label="My Money", color="blue")
ax.plot(df["date"], df["baseline_money"], label="Baseline Money", color="red")

# Horizontal line at 10,000
ax.axhline(y=10_000, color="black", linestyle="--", linewidth=1)

# Vertical line at the beginning of every year
for year in df["date"].dt.year.unique():
  ax.axvline(
    pd.Timestamp(f"{year}-01-01"),
    color="gray",
    linestyle="--",
    linewidth=0.8,
  )

ax.set_xlabel("Date")
ax.set_ylabel("Money")
ax.set_title("My Money vs Baseline Money")
ax.legend()

plt.tight_layout()
plt.savefig(f"../_plots/{RUN_ID}.png")