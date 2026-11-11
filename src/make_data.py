"""Makes fake sensor readings for a fleet of machines.

Each row is one reading, and `fault` says whether the machine failed within
the next few days. Failures are rare and tend to follow high vibration, high
temperature or a long time since the last service, but with plenty of noise.
For real data, save a CSV as data/readings.csv with numeric columns and a
`fault` column (0 or 1) and skip this step.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)


def main(n=15000):
    df = pd.DataFrame({
        "temperature": rng.normal(70, 8, n),
        "vibration": rng.gamma(2.0, 1.2, n),
        "pressure": rng.normal(30, 4, n),
        "rpm": rng.normal(1500, 120, n),
        "hours_since_service": rng.integers(0, 2000, n),
    })

    # a made-up "risk" score. Faults are more likely when it is high
    risk = (
        0.08 * (df["temperature"] - 70)
        + 0.6 * (df["vibration"] - 2.4)
        + 0.002 * (df["hours_since_service"] - 1000)
        + 0.004 * (df["rpm"] - 1500) / 10
    )
    prob = 1 / (1 + np.exp(-(risk * 1.5 - 4.2)))
    df["fault"] = rng.binomial(1, prob)

    # real sensors drop out sometimes
    for col in ["temperature", "vibration", "pressure"]:
        missing = rng.random(n) < 0.02
        df.loc[missing, col] = np.nan

    df.to_csv("data/readings.csv", index=False)
    print("saved", len(df), "rows, fault rate", round(df["fault"].mean(), 3))


if __name__ == "__main__":
    main()
