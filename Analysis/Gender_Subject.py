"""
Subject Gender Subgroup Analysis for STEMorph

This script performs a gender-based subgroup analysis, comparing emotional ratings between male and female subjects. It processes
participant ratings, performs outlier removal, conducts linear regression,
and generates visualizations for gender comparisons.

Author: Mohammad Ebrahim Katebi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn as sns
import pingouin as pg

# Constants
DATA_DIR = '../Data_Validity/'
RESULTS_DIR = '../Results_Gender/'
GENDER_FILE = '../Subject_Gender.csv'


def load_and_preprocess_data(file_name):
    """
    Load and preprocess data from a CSV file.

    Args:
        file_name (str): Name of the CSV file to load.

    Returns:
        pandas.DataFrame: Preprocessed data table.
    """
    data = pd.read_csv(os.path.join(DATA_DIR, file_name), header=0)
    data = data[data['State'] == 1]
    table = data.drop(columns=['ID', 'State', 'Position ID', 'Face Person',
                               'ITI', 'Trial_Onset', 'Stim_Onset', 'Stim_Offset'])
    return table


def remove_outliers(table):
    """
    Remove outliers from the data.

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


def create_gender_comparison_plot(table, file_name):
    """
    Create and save a gender comparison plot for the given data.

    Args:
        table (pandas.DataFrame): Preprocessed data table.
        file_name (str): Name of the file for saving the plot.
    """
    plt.figure()
    sns.set_style("ticks")
    sns.despine()

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    palette = sns.color_palette("Spectral", 2)

    # Draw violin plot
    sns.violinplot(data=table, x="Morph Step", y="Answer", hue="Gender", palette=palette,
                   split=True, gap=.45, inner=None, cut=3, native_scale=True, linewidth=0.8, width=1.1)

    title = ""
    for gender in ['Male Subject', 'Female Subject']:
        gender_data = table[table['Gender'] == gender]
        x = gender_data['Morph Step'].values.reshape(-1, 1)
        y = gender_data['Answer']

        regr, intercept, slope = perform_linear_regression(x, y)
        r_squared = round(pg.corr(gender_data['Morph Step'], y)[
                          'r'].values[0] ** 2, 3)

        color = '#C0DCA9' if gender == 'Male Subject' else '#ECBD81'
        sns.lineplot(data=gender_data, x="Morph Step", y=regr.predict(x),
                     color=color, linewidth=1, linestyle='dotted')

        morph_steps = gender_data.groupby('Morph Step')['Morph Step'].mean()
        means = gender_data.groupby('Morph Step')['Answer'].mean()
        plt.scatter(x=morph_steps, y=means, color=color, s=16)

        title += f'{gender}s: R2 = {r_squared} | y = {slope} . x + {intercept}\n'

    plt.title(title, fontsize=10, loc='center')
    plt.xlabel('Morphing Step')
    plt.ylabel('Answer')
    plt.xticks(np.arange(1, 10, 1))
    plt.yticks(np.arange(1, 10, 1))

    # Remove top and right frames
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save plot
    plt.savefig(os.path.join(RESULTS_DIR, f'{file_name} Regression.png'), dpi=1000,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    """
    Main function to run the subject gender subgroup analysis.
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load gender information
    genders = pd.read_csv(GENDER_FILE, header=0)

    # Process all files
    list_of_files = os.listdir(DATA_DIR)
    table_all = pd.DataFrame()

    for file_name in list_of_files:
        print(f"Processing {file_name}")
        participant_id = int(file_name[8:12])
        gender = genders[genders['ID'] == participant_id]['Gender'].iloc[0]

        table = load_and_preprocess_data(file_name)
        table['Gender'] = f'{gender} Subject'
        table_all = pd.concat([table_all, table], axis=0)

    # Remove outliers and create integrated plot
    table_all = remove_outliers(table_all)
    create_gender_comparison_plot(table_all, 'Gender_Subject Subject Average')


if __name__ == "__main__":
    main()
