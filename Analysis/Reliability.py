"""
Reliability Analysis for STEMorph

This script performs reliability analysis on STEMorph, comparing initial and retest
participant ratings. It processes data from both sessions, performs outlier removal, conducts linear regression, and generates visualizations.

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
RELIABILITY_FILES_ADDRESS = '../Data_Reliability/'
VALIDITY_FILES_ADDRESS = '../Data_Validity/'
RESULTS_DIR = '../Results_Reliability/'


def load_and_preprocess_data(file_name, address):
    """
    Load and preprocess data from a CSV file.

    Args:
        file_name (str): Name of the CSV file to load.
        address (str): Directory containing the file.

    Returns:
        pandas.DataFrame: Preprocessed data table.
    """
    data = pd.read_csv(os.path.join(address, file_name), header=0)
    data = data[data['State'] == 1]
    table = data.drop(
        columns=['State', 'ITI', 'Trial_Onset', 'Stim_Onset', 'Stim_Offset', 'RT'])
    return table


def remove_outliers(table):
    """
    Remove outliers from the data.

    Args:
        table (pandas.DataFrame): Input data table.

    Returns:
        pandas.DataFrame: Data table with outliers removed.
    """
    for answer_validity in np.unique(table['Answer_Validity']):
        step_data = table[table['Answer_Validity'] == answer_validity]
        mean = step_data['Answer'].mean()
        std = step_data['Answer'].std()
        table = table[
            (table['Answer_Validity'] != answer_validity) |
            ((table['Answer_Validity'] == answer_validity) &
             (abs(table['Answer'] - mean) < 2 * std))
        ]
    return table


def perform_linear_regression(x, y):
    """
    Perform linear regression.

    Args:
        x (numpy.ndarray): Input features (Initial answers).
        y (numpy.ndarray): Target variable (Retest answers).

    Returns:
        tuple: Regression model, intercept, and slope.
    """
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    intercept = round(regr.intercept_, 3)
    slope = round(regr.coef_[0], 3)
    return regr, intercept, slope


def create_reliability_plot(table, file_name):
    """
    Create and save a reliability plot for the given data.

    Args:
        table (pandas.DataFrame): Preprocessed data table.
        file_name (str): Name of the file for saving the plot.
    """
    # Create a copy of the table for violin plots with outliers removed
    table_no_outliers = remove_outliers(copy.deepcopy(table))

    x = table['Answer_Validity'].values.reshape(-1, 1)
    y = table['Answer']

    regr, intercept, slope = perform_linear_regression(x, y)
    r_squared = round(pg.corr(table['Answer_Validity'], y)[
                      'r'].values[0] ** 2, 3)

    plt.figure(figsize=(9, 9))
    sns.set_style("ticks")
    sns.despine()

    # Remove top and right frames
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xticks(range(0, 9))
    plt.yticks(range(1, 10))
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    palette = sns.color_palette(
        "vlag", np.unique(table['Answer_Validity']).size)
    palette.reverse()

    # Plot data
    sns.violinplot(data=table_no_outliers, x="Answer_Validity", y="Answer", hue="Answer_Validity", legend=False,
                   inner=None, cut=3, native_scale=True, palette=palette, saturation=1,
                   linewidth=0.8, width=0.6, fill=True, alpha=0.85)

    # Calculate jitter
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    aspect_ratio = bbox.width / bbox.height
    angle = np.random.uniform(0, 2 * np.pi, table['Answer_Validity'].size)
    radius = np.random.uniform(0, 0.1, table['Answer_Validity'].size)
    x_jitter = radius * np.cos(angle)
    y_jitter = radius * np.sin(angle) / (1 / aspect_ratio)

    # Apply jitter and plot
    table['Answer_Validity_Jittered'] = table['Answer_Validity'] + x_jitter
    table['Answer_Jittered'] = table['Answer'] + y_jitter
    sns.scatterplot(data=table, x="Answer_Validity_Jittered", y="Answer_Jittered",
                    s=1, color='black', marker=".", linewidth=0.05)

    sns.lineplot(data=table, x="Answer_Validity", y=regr.predict(x),
                 color='gray', linewidth=1, linestyle='dotted')

    answers_validity = table.groupby('Answer_Validity')[
        'Answer_Validity'].mean()
    means = table.groupby('Answer_Validity')['Answer'].mean()
    plt.scatter(x=answers_validity, y=means, color='black', s=19)

    # Labels and title
    plt.xlabel('Validity Rating', fontsize=12)
    plt.ylabel('Retest Answers', fontsize=12)
    plt.xticks(np.arange(1, 10, 1))
    plt.yticks(np.arange(1, 10, 1))
    plt.title(f'Reliability - Subject Average\nR2 = {r_squared} | Particiapans\' Retest Rating = {intercept} + {slope} . VR',
              fontsize=12)

    # Save plot
    plt.savefig(os.path.join(RESULTS_DIR, f'{file_name} Regression.png'), dpi=400,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    """
    Main function to run the reliability analysis.
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process all files
    reliability_files = os.listdir(RELIABILITY_FILES_ADDRESS)
    validity_files = os.listdir(VALIDITY_FILES_ADDRESS)
    table_all = pd.DataFrame()

    for reliability_file in reliability_files:
        print(f"Processing {reliability_file}")
        participant_id = reliability_file[8:12]
        reliability_table = load_and_preprocess_data(
            reliability_file, RELIABILITY_FILES_ADDRESS)

        # Find corresponding validity file
        validity_file = next(
            (f for f in validity_files if participant_id in f), None)
        if validity_file:
            print(f"Matched validity file: {validity_file}")
            validity_table = load_and_preprocess_data(
                validity_file, VALIDITY_FILES_ADDRESS)

            # Merge reliability and validity data
            merged_table = pd.merge(reliability_table, validity_table,
                                    on=['Position ID',
                                        'Face Person', 'Morph Step'],
                                    suffixes=['', '_Validity'])
            table_all = pd.concat([table_all, merged_table], axis=0)
        else:
            print(f"No matching validity file found for {participant_id}")

    # Process combined data
    # table_all = remove_outliers(table_all)
    create_reliability_plot(table_all, 'Subject Average')


if __name__ == "__main__":
    main()
