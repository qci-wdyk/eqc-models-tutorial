# (C) Quantum Computing Inc., 2024.
import os
import datetime
import numpy as np
import pandas as pd

NASDAQ100_TABLE = (
    "nasdaq100_stocks_table.csv"
)
DROP_STOCKS = []
STOCK_DATA_DIR = "stock_prices"


def remove_unavailable_stocks(
    stocks, adj_date, lookback_days=60, lookforward_days=30
):
    sel_stocks = []
    for stock in stocks:
        stock_df = pd.read_csv(
            os.path.join(STOCK_DATA_DIR, "%s.csv" % stock)
        )

        if stock_df.shape[0] == 0:
            continue

        stock_df["Date"] = stock_df["Date"].astype("datetime64[ns]")

        adj_date = pd.to_datetime(adj_date)
        beg_date = adj_date - datetime.timedelta(days=lookback_days)
        end_date = adj_date + datetime.timedelta(days=lookforward_days)

        stock_df = stock_df[
            (stock_df["Date"] >= beg_date) & (stock_df["Date"] <= end_date)
        ]

        if stock_df.shape[0] == 0:
            continue

        sel_stocks.append(stock)

    print("Chose %d of %d stocks" % (len(sel_stocks), len(stocks)))

    return sel_stocks


def get_nasdaq100_constituents(
    adj_date, lookback_days=60, lookforward_days=30
):
    adj_date = pd.to_datetime(adj_date)

    df = pd.read_csv(NASDAQ100_TABLE, low_memory=False)

    df["beg_date"] = df["beg_date"].apply(pd.to_datetime)

    beg_dates = sorted(df["beg_date"].unique(), reverse=True)

    for beg_date in beg_dates:
        if adj_date >= beg_date:
            break

    stocks = list(df[df["beg_date"] == beg_date]["symbol"].unique())
    stocks = list(set(stocks) - set(DROP_STOCKS))
    stocks = remove_unavailable_stocks(
        stocks, adj_date, lookback_days, lookforward_days
    )

    return stocks


def calc_port_vals(df, init_port_val, out_of_sample_days, mode="unequal"):
    df["Date"] = df["Date"].astype("datetime64[ns]")
    beg_port_val = init_port_val
    df = df.sort_values("Date")
    adj_dates = sorted(df["Date"].unique())
    num_adj_dates = len(adj_dates)
    dates = None
    port_vals = None
    for i in range(num_adj_dates):
        beg_date = pd.to_datetime(adj_dates[i])

        print("Processing %s..." % beg_date.strftime("%Y-%m-%d"))

        if i < num_adj_dates - 1:
            end_date = pd.to_datetime(adj_dates[i + 1])
        else:
            end_date = beg_date + datetime.timedelta(
                days=out_of_sample_days
            )

        tmp_df = df[df["Date"] == beg_date]
        stocks = tmp_df["Stock"]

        stocks = list(
            set(stocks) - {"FISV", "RE", "ABC", "PKI", "SIVB", "FRC"}
        )

        if end_date > pd.to_datetime("2023-10-20"):
            stocks = list(set(stocks) - {"ATVI"})

        if beg_date < pd.to_datetime("2008-04-23"):
            stocks = list(set(stocks) - {"AWK"})

        if end_date > pd.to_datetime("2022-12-23"):
            stocks = list(set(stocks) - {"ABMD"})

        # if (
        #         beg_date >= pd.to_datetime("2012-12-01")
        #         and end_date >= pd.to_datetime("2013-01-01")
        # ):
        #     stocks = list(set(stocks) - {"BMC"})

        all_dates = [beg_date]
        date0 = beg_date
        while date0 < end_date:
            date0 = date0 + datetime.timedelta(days=1)
            all_dates.append(date0)

        price_df = pd.DataFrame({"Date": all_dates})
        for stock in stocks:
            stock_df = pd.read_csv(
                os.path.join(STOCK_DATA_DIR, "%s.csv" % stock)
            )
            stock_df["Date"] = stock_df["Date"].astype("datetime64[ns]")

            stock_df = stock_df[
                (stock_df["Date"] >= beg_date)
                & (stock_df["Date"] <= end_date)
            ]

            assert stock_df.shape[0] > 0, "No data for %s, %s, %s" % (
                stock,
                str(beg_date),
                str(end_date),
            )

            if price_df is None:
                price_df = stock_df
            else:
                price_df = price_df.merge(stock_df, on="Date", how="outer")

        price_df = price_df.fillna(method="ffill").fillna(method="bfill")
        price_df = price_df.sort_values("Date")

        tmp_dates = np.array(price_df["Date"])
        tmp_port_vals = np.zeros(shape=(price_df.shape[0]))

        sum_wt = 0.0
        if mode != "equal":
            for stock in stocks:
                stock_wt = float(
                    tmp_df[tmp_df["Stock"] == stock]["Allocation"].item()
                )
                sum_wt += stock_wt

        for stock in stocks:
            try:
                prices = np.array(price_df[stock])
            except Exception as exc:
                print(exc)
                print(price_df)
                sys.exit()

            beg_price = prices[0]

            if mode == "equal":
                stock_wt = 1.0 / len(stocks)
            else:
                stock_wt = (
                    float(
                        tmp_df[tmp_df["Stock"] == stock][
                            "Allocation"
                        ].item()
                    )
                    / sum_wt
                )

            stock_count = stock_wt * beg_port_val / beg_price

            tmp_port_vals += stock_count * prices

        if dates is None:
            dates = tmp_dates
        else:
            dates = np.concatenate([dates, tmp_dates])

        if port_vals is None:
            port_vals = tmp_port_vals
        else:
            port_vals = np.concatenate([port_vals, tmp_port_vals])

        beg_port_val = port_vals[-1]

    return dates, port_vals


def get_port_stats(weight_df, lookforward_days):
    dates, vals = calc_port_vals(weight_df, 1.0, lookforward_days)

    df = pd.DataFrame({"Date": dates, "port_val": vals})
    df = df.sort_values("Date", ascending=True)
    df["daily_return"] = df["port_val"].pct_change()

    mean_port_ret = df["daily_return"].mean()
    std_port_ret = df["daily_return"].std()

    ind_df = pd.read_csv(os.path.join(STOCK_DATA_DIR, "^NDX.csv"))
    ind_df["Date"] = ind_df["Date"].astype("datetime64[ns]")
    adj_date = list(weight_df["Date"])[0]
    min_date = pd.to_datetime(adj_date)
    max_date = min_date + datetime.timedelta(days=lookforward_days)
    ind_df = ind_df[
        (ind_df["Date"] >= min_date) & (ind_df["Date"] <= max_date)
    ]
    ind_df["daily_return"] = ind_df["^NDX"].pct_change()

    mean_ind_ret = ind_df["daily_return"].mean()
    std_ind_ret = ind_df["daily_return"].std()

    ret_df = pd.DataFrame(
        {
            "Portfolio": ["Optimized Portfolio", "Nasdaq-100"],
            "Avg. Daily Return (%)": [
                100.0 * mean_port_ret,
                100.0 * mean_ind_ret,
            ],
            "Std. Dev. Daily Return (%)": [
                100.0 * std_port_ret,
                100.0 * std_ind_ret,
            ],
        }
    )

    return ret_df
