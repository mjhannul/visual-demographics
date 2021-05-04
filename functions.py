"""
This module contains auxiliary functions for the notebook.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def country_to_code(country, dictionary):
    """Find country code using country name."""
    for key, item in dictionary.items():
        if item == country:
            return key
    return None


def code_to_country(country_code, dictionary):
    """Find country name using country code."""
    for key, item in dictionary.items():
        if key == country_code:
            return item
    return None


def join_data(dictionary, data, year_list):
    """
    Join several heterogeneous datasets.

    Positional arguments:
    :param dictionary: dictionary where values are datasets whose rows represent countries and columns
    :param data: dataset whose rows represent country-year pairs
    :param years: year range for output dataset (inclusive)py
    """
    for key, frame in dictionary.items():
        data["temp"] = np.nan  # Initialize temporary column 'temp'
        for _, row in frame.iterrows():
            rowcode = row.loc["country_code"]
            years = range(year_list[0], year_list[1] + 1)
            dict_temp = {
                "country_code": [rowcode for year in years],
                "year": list(years),
                key: [row.loc[str(year)] for year in years],
            }
            dftemp = pd.DataFrame(
                data=dict_temp
            )  # Create a temporary data frame for one country
            data = pd.merge(
                data, dftemp, how="outer"
            )  # Join with data by outer join (because data['temp'] is constant null)
            data["temp"].fillna(
                data[key], inplace=True
            )  # Copy added values to column 'temp'
            data.drop(columns=[key], inplace=True)
        data[key] = data["temp"]  # Copy column 'temp' to column key
        data.drop(columns=["temp"], inplace=True)  # Drop 'temp' column
    return data


def plot_country_bars(data, year, stat):
    """Create bar plot for country stats."""
    if data.size != 0:
        sns.set_theme(style="whitegrid")
        _, ax = plt.subplots(figsize=(6, 55))
        mypalette = sns.light_palette("seagreen", data.size)
        sns.barplot(x=stat, y="country_name", data=data, palette=mypalette[::-1])
        ax.set(
            xlabel=f"{stat} in {year}",
            ylabel="Country",
            title=f"{stat} in {year}",
        )
        for patch in ax.patches:
            width = patch.get_width()
            height = patch.get_height()
            plt.text(
                width + 0.01 * data[stat].max(),
                patch.get_y() + 0.55 * height,
                "{:1.2f}".format(width),
                ha="left",
                va="center",
            )
    else:
        print("\n\nThe dataset is empty!")


def country_stats(country_code, data):
    """Fetch data for a selected country code."""
    data = data[data.country_code == country_code].sort_values(by=["Year"])
    data.name = country_code
    data.set_index("Year", inplace=True)
    return data


def wavg(group, metric_name, weight_name):
    """Return weighted average for continent stats (https://pbpython.com/weighted-average.html)"""
    metric = group[metric_name]
    weight = group[weight_name]
    return (metric * weight).sum() / weight.sum()


def continent_stats(cont_code, data, stat):
    """Aggregate continents stats w.r.t. population"""
    # Remove pairs with missing values in stat and "Population"
    data1 = data.dropna(subset=[stat, "Population"])
    # Group by cont_code, year and compute weighted average of stat
    groups = data1.groupby(["cont_code", "Year"]).apply(wavg, stat, "Population")[
        cont_code
    ]
    return groups


def plot_area(data, code, metrics, code_dicts, years):
    """
    Create two-line plot for continent/country stats

    Positional arguments:
    :param data: dataframe containing country stats
    :param code: country or continent code
    :param metrics: list of two metrics to be plotted
    :param code_dicts: list of country and continent code dictionaries
    :param years: list of start and end year
    """
    fig = plt.figure(figsize=(9, 5))
    if code in code_dicts[0].keys():
        data1 = country_stats(code, data)[metrics[0]]
        data2 = country_stats(code, data)[metrics[1]]
        plt.title(
            f"{metrics[0]} and {metrics[1]} in {code_dicts[0][code]} in {years[0]}-{years[1]}",
            loc="center",
        )
    elif code in code_dicts[1].keys():
        data1 = continent_stats(code, data, metrics[0])
        data2 = continent_stats(code, data, metrics[1])
        plt.title(
            f"{metrics[0]} and {metrics[1]} in {code_dicts[1][code]} in {years[0]}-{years[1]}",
            loc="center",
        )
    ax = fig.add_subplot(111)
    ax.grid(False)
    plt.xlabel("Year")
    plt.ylabel(metrics[0])
    sns.lineplot(data=data1, color="r", label=metrics[0])
    ax2 = ax.twinx()
    ax2.grid(False)
    sns.lineplot(data=data2, color="b", label=metrics[1])
    plt.ylabel(metrics[1])
    fig.legend(loc="lower left", bbox_to_anchor=(0, -0.1))
    ax.get_legend().remove()
    ax2.get_legend().remove()
    plt.show()


def plot_multi_predictors(data, response, predictors):
    """
    Create plots for one response and different predictor variables

    Positional variables:
    :param data: dataframe
    :param response: plotted response variable
    :param predictors: list of predictor variables
    """
    length = len(predictors)
    nrows = int(length / 3) + 1
    _, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.4)
    for i in range(length):
        sns.regplot(
            y=response,
            x=predictors[i],
            data=data,
            ax=axes[int(i / 3), i % 3],
            marker=".",
        )
        plt.xlabel(predictors[i])
        plt.ylabel(response)
    for j in range(length % 3, 3):
        axes[int(length / 3), j].set_axis_off()


def plot_world_map(world_df, year, response, quantile_number):
    """
    Create a world map visualization

    Positional variables:
    :param world_df: geodataframe
    :param year: integer
    :param response: plotted variable
    :param quantile_number: number of colours in visualization
    """
    world_df_year = world_df[world_df.Year == year]
    _, ax = plt.subplots(1, figsize=(35, 35))
    # Use geopandas plot method
    world_df_year.plot(
        column=response,
        cmap="Greens",
        linewidth=1,
        ax=ax,
        edgecolors="0.8",
        scheme="Quantiles",
        k=quantile_number,
        legend=True,
        legend_kwds={
            "fmt": "{:.1f}",
            "fontsize": 25,
            "markerscale": 2.5,
            "loc": "lower left",
        },
        missing_kwds={"color": "lightgrey", "label": "No Data"},
    )
    ax.set_title(f"{response} in {year}", fontdict={"fontsize": 30})
    ax.set_axis_off()
