#!/usr/bin/env python

# Contains functions for plotting well data in various ways

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def iqr_filter(data):
    """
    Takes a DataFrame containing well log data and filters outliers using Interquartile Range

    Parameters:
    data, pandas.DataFrame
        The input data to be filtered.  Each column should be a well log curve

    Returns:
    df_filt, pandas.DataFrame
        The filtered well curves
    """
    # create a copy of the original data
    df_copy = data.copy()
    # create a data frame containing the cleaned version of the data using IQR
    df_filt = pd.DataFrame(index=df_copy.index)
    # filter each column
    for col in df_copy.columns.tolist():
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        IQR = q3 - q1
        iqr_filt = (df_copy[col] >= q1 - 1.5 * IQR) & (df_copy[col] <= q3 + 1.5 * IQR)
        df_filt[col] = df_copy[col].loc[iqr_filt]
    return df_filt


def plot_distributions(df):
    """
    Plots histogram for each curve in df
    Args:
        df: pandas.DataFrame

    Returns:
        matplotlib.pyplot.figure.Figure
    """
    flag = 0
    n_blank = 0
    n_subplots = len(df.columns.tolist())
    if n_subplots % 2 == 0:
        n_rows = 2
        n_cols = n_subplots // n_rows
    elif n_subplots % 3 == 0:
        n_rows = 3
        n_cols = n_subplots // n_rows
    else:
        n_cols = n_subplots // 2
        n_rows = n_subplots // n_cols
        n_rows += n_subplots % n_cols
        n_blank = (n_rows * n_cols) - n_subplots
        flag = 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    if flag == 1:
        axes[-1, -n_blank].axis('off')
    for ax, col in zip(axes.flatten(), df.columns.tolist()):
        ax.hist(df[col])
        if 'HR' in col:
            ax.set_xscale('log')
        ax.set_title('{} Histogram'.format(col))
    plt.tight_layout()
    return fig


def plot_well_curves(df, hist=True):
    """
    Plots well curves from an input data frame containing data
    Args:
        df: pandas.DataFrame,
            containing well logs, one log per column

        hist: bool,
            Whether or not to plot curve histogram at bottom of log plot

    Returns:
        matplotlib.pyplot.figure.Figure

    """
    # get the column names as a list
    curve_names = df.columns.tolist()

    # check whether plots should include a histogram
    n_rows = 1
    if hist:
        n_rows = 2

    # create the figure
    if hist:
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=len(curve_names),
            sharey='row',
            figsize=(20, 12),
            gridspec_kw={'height_ratios': [4, 1]}
        )
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=len(curve_names), sharey='row', figsize=(28, 10))
    fig.suptitle('Well Log Panel', fontsize=20)
    if hist:
        num_plots = n_rows * len(curve_names)
        start_row_two = int(num_plots / 2)
        curves = df.columns.tolist()
        for i, ax in enumerate(axes.flatten()):
            if i < start_row_two:
                if curves[i] in ['HRD', 'HRM']:
                    ax.semilogx(df[curves[i]], df.index, color='k')
                else:
                    ax.plot(df[curves[i]], df.index, color='k')
                if curves[i] == 'CNC':
                    ax.set_xlim(0.0, 1.0)
                if curves[i] in ['DTC', 'DTS'] or 'DT' in curves[i]:
                    ax.set_title(curves[i], fontdict={'color': 'r'})
                    ax.invert_xaxis()
                else:
                    ax.set_title(curves[i])
                ax.xaxis.tick_top()
            else:
                ax.hist(df[curves[i - start_row_two]].dropna())
    else:
        for ax, curve in zip(axes, curve_names):
            if curve in ['HRD', 'HRM']:
                ax.semilogx(df[curve], df.index, color='k')
            else:
                ax.plot(df[curve], df.index, color='k')
            if curve == 'CNC':
                ax.set_xlim(0.0, 1.0)
            if curve in ['DTC', 'DTS'] or 'DT' in curve:
                ax.set_title(curve, fontdict={'color': 'r'})
                ax.invert_xaxis()
            else:
                ax.set_title(curve)
            ax.xaxis.tick_top()
    axes[0, 0].invert_yaxis()  # using sharey=True will invert y-axis for all plots
    axes[0, 0].set_ylabel('Depth', fontdict={'fontsize': 20})
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_vp_vs(x='DTC', y='DTS', color='index', df=None):
    """
    Plots measured Vp versus Vs against well-known rock physics trends
    Args:
        x: (pandas.Series) input DTC values in us/ft
        y: (pandas.Series) input DTS values in us/ft
        color: (str) Input variable for coloring scatter plots.  Acceptable
        values are 'index' or any of the input data frame column names.
        df: (pandas.DataFrame) input data frame containing x & y

    Returns:
        matplotlib.pyplot.figure.Figure
    """

    vp_ft_s = 1e6 / df[x]
    vp_m_s = vp_ft_s / 3.281
    vp_km_s = vp_m_s / 1000.

    vs_ft_s = 1e6 / df[y]
    vs_m_s = vs_ft_s / 3.281
    vs_km_s = vs_m_s / 1000.

    xvp = np.arange(start=0, stop=8, step=0.1)

    df_keys = df.columns.tolist()
    color_mapping = dict()
    color_mapping['index'] = df.index

    for key in df_keys:
        if key not in color_mapping:
            color_mapping[key] = df[key]

    Vs_castagna_ls = np.multiply(-0.05508, np.power(xvp, 2)) + np.multiply(1.0168, xvp) - 1.0305
    Vs_castagna_dm = np.multiply(0.5832, xvp) - 0.07776
    Vs_castagna_mudrock = np.multiply(0.8621, xvp) - 1.1724
    Vs_castagna_ss = np.multiply(0.8042, xvp) - 0.8559

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Vs_castagna_ls, xvp, '--b', lw=2, label='Castagna et al. (1993) Water-Saturated Limestone')
    ax.plot(Vs_castagna_dm, xvp, '--c', lw=2, label='Castagna et al. (1993) Water-Saturated Dolomite')
    ax.plot(Vs_castagna_mudrock, xvp, '--k', lw=2, label='Castagna et al. (1993) Mudrock line')
    ax.plot(Vs_castagna_ss, xvp, '--r', lw=2, label='Castagna et al. (1993) Water-Saturated Stone')
    if color in ['GR', 'ZDEN', 'CNC', 'PE']:
        im = ax.scatter(vs_km_s, vp_km_s, s=10, c=color_mapping[color], marker='.', cmap='inferno_r', alpha=0.8)
    else:
        im = ax.scatter(vs_km_s, vp_km_s, s=10, c=color_mapping[color], marker='.', cmap='inferno', alpha=0.8)
    ax.set_xlabel('Vs (km/s)')
    ax.set_ylabel('Vp (km/s)')
    ax.set_title('Vp vs. Vs', fontsize=20, fontweight='bold')
    ax.set_xlim([np.nanmin(vs_km_s), np.nanmax(vs_km_s)])
    ax.set_ylim([np.nanmin(vp_km_s), np.nanmax(vp_km_s)])
    ax.legend(loc='upper left')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.invert_yaxis()
    cbar.set_label(f'{color}', fontdict={'fontweight': 'bold'})

    return fig
