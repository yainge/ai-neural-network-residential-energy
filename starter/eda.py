"""Exploratory data analysis for the Residential Energy Challenge.

Kept separate from dataset.py so training and evaluation never block on plots.
Run: python starter/eda.py
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from dataset import load_and_enrich


def main() -> None:
    df = load_and_enrich()

    # check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("Missing values per column:")
        print(nan_counts[nan_counts > 0])

    # dataset description
    print(df.describe())

    # plots for visual data inspection
    hourly = df.set_index('timestamp')['kwh']
    daily = hourly.resample('D').sum()

    # Hourly consumption
    hourly.plot(figsize=(14, 4), title='Hourly Energy Consumption')
    plt.ylabel('kWh')
    plt.show()

    # Daily totals
    daily.plot(figsize=(14, 4), title='Daily Energy Consumption')
    plt.ylabel('kWh per day')
    plt.show()

    # Weekdays only (Mon–Fri)
    weekdays = daily[daily.index.dayofweek < 5]
    weekdays.plot(figsize=(14, 4), marker='.', title='Daily Energy Consumption: Weekdays (Mon–Fri)')
    plt.ylabel('kWh per day')
    plt.show()

    # Weekend days only (Sat–Sun)
    weekends = daily[daily.index.dayofweek >= 5]
    weekends.plot(figsize=(14, 4), marker='.', color='tab:orange', title='Daily Energy Consumption: Weekends (Sat–Sun)')
    plt.ylabel('kWh per day')
    plt.show()


if __name__ == "__main__":
    main()
