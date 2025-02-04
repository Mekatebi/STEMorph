"""
Validity Analysis for STEMorph

This script performs validity analysis on STEMorph, a set of morphed
emotional face stimuli ranging from angry to happy. It processes participant ratings,
performs outlier removal, conducts linear regression, and generates visualizations.

Author: Mohammad Ebrahim Katebi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn as sns
import pingouin as pg
import copy

# Constants
FILES_ADDRESS = '../Data_Validity/'
RESULTS_DIR = '../Results_Validity/'


def load_and_preprocess_data(file_name):
    """
    Load and preprocess data from a CSV file.

    Args:
        file_name (str): Name of the CSV file to load.

    Returns:
        pandas.DataFrame: Preprocessed data table.
    """
    data = pd.read_csv(os.path.join(FILES_ADDRESS, file_name), header=0)
    data = data[data['State'] == 1]
    table = data.drop(columns=['State', 'Position ID', 'Face Person', 'ITI',
                               'Trial_Onset', 'Stim_Onset', 'Stim_Offset'])
    return table


def remove_outliers(table):
    """
    Remove outliers from the data

    Args:
        table (pandas.DataFrame): Input data table.

    Returns:
        pandas.DataFrame: Data table with outliers removed.
    """
    for morph_step in np.unique(table['Morph Step']):
        step_data = table[table['Morph Step'] == morph_step]
        mean = step_data['Answer'].mean()
        std = step_data['Answer'].std()
        table = table[
            (table['Morph Step'] != morph_step) |
            ((table['Morph Step'] == morph_step) &
             (abs(table['Answer'] - mean) < 2 * std))
        ]
    return table


def perform_linear_regression(x, y):
    """
    Perform linear regression on the input data.

    Args:
        x (numpy.ndarray): Input features (Morph Step).
        y (numpy.ndarray): Target variable (Answer).

    Returns:
        tuple: Regression model, intercept, and slope.
    """
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    intercept = round(regr.intercept_, 3)
    slope = round(regr.coef_[0], 3)
    return regr, intercept, slope


def create_validity_plot(table, file_name):
    """
    Create and save a validity plot for the given data.

    Args:
        table (pandas.DataFrame): Preprocessed data table.
        file_name (str): Name of the file for saving the plot.
    """
    # Create a copy of the table for violin plots with outliers removed
    table_no_outliers = remove_outliers(copy.deepcopy(table))

    x = table['Morph Step'].values.reshape(-1, 1)
    y = table['Answer']

    regr, intercept, slope = perform_linear_regression(x, y)
    r_squared = round(pg.corr(table['Morph Step'], y)['r'].values[0] ** 2, 3)

    plt.figure()
    sns.set_style("ticks")
    sns.despine()
    # Remove top and left frames:
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot setup
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    palette = sns.color_palette(
        "coolwarm", np.unique(table['Morph Step']).size)
    palette.reverse()

    # Calculate jitter
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    aspect_ratio = bbox.width / bbox.height
    angle = np.random.uniform(0, 2 * np.pi, table['Morph Step'].size)
    radius = np.random.uniform(0, 0.12, table['Morph Step'].size)
    x_jitter = radius * np.cos(angle)
    y_jitter = radius * np.sin(angle) / (1 / aspect_ratio)

    # Apply jitter and plot
    table['Morph_Step_Jittered'] = table['Morph Step'] + x_jitter
    table['Answer_Jittered'] = table['Answer'] + y_jitter
    sns.scatterplot(data=table, x="Morph_Step_Jittered", y="Answer_Jittered",
                    s=1, color='black', marker=".")

    sns.violinplot(data=table_no_outliers, x="Morph Step", y="Answer", hue="Morph Step", legend=False,
                   inner=None, cut=3, native_scale=True, palette=palette, saturation=1,
                   linewidth=0.8, width=0.6, fill=False)

    sns.lineplot(data=table, x="Morph Step", y=regr.predict(x),
                 color='gray', linewidth=1, linestyle='dotted')

    morph_steps = table.groupby('Morph Step')['Morph Step'].mean()
    means = table.groupby('Morph Step')['Answer'].mean()
    plt.scatter(x=morph_steps, y=means, color='black', s=18)

    # Labels and title
    plt.xlabel('Morphing Step', fontsize=9)
    plt.ylabel('Answer', fontsize=9)
    plt.xticks(np.arange(1, 10, 1))
    plt.yticks(np.arange(1, 10, 1))
    plt.title(
        f'{file_name}\nR2 = {r_squared} | y = {slope} . x + {intercept}', fontsize=9)

    # Save plot
    plt.savefig(os.path.join(RESULTS_DIR, f'{file_name} Regression.png'), dpi=600,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    """
    Main function
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process all files
    list_of_files = os.listdir(FILES_ADDRESS)
    table_all = pd.DataFrame()

    for file_name in list_of_files:
        table = load_and_preprocess_data(file_name)
        table_all = pd.concat([table_all, table], axis=0)
        # Uncomment the following line to process individual files
        # create_validity_plot(remove_outliers(table), file_name)
        print(f"Processed {file_name}")

    # Process combined data
    # table_all = remove_outliers(table_all)
    create_validity_plot(table_all, 'Validity - Subject Average')


if __name__ == "__main__":
    main()
