"""
Validity Analysis for STEMorph with Gender Subgroups

This script performs validity analysis on STEMorph stimuli with detailed gender subgroup analysis.
Uses the same full model as the main validity analysis and creates separate visualizations
for face and subject gender subgroups.

Author: Mohammad Ebrahim Katebi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Constants
FILES_ADDRESS = '../Data_Validity/'
RESULTS_DIR = '../Results_gender/'
GENDER_FILE = '../Subject_Gender.csv'
PALETTE_SUBJECT = {'Female': '#2596be', 'Male': '#be4d25'}
PALETTE_FACE = {'Female': '#00cec9', 'Male': '#fdcb6e'}


def validate_data(table):
    """Validate data table structure and content."""
    required_columns = ['ID', 'Face Person', 'Morph Step', 'Answer']

    if not all(col in table.columns for col in required_columns):
        print("Error: Missing required columns")
        return False

    if table.empty:
        print("Error: Empty data table")
        return False

    if table['Morph Step'].isnull().any() or table['Answer'].isnull().any():
        print("Error: Missing values in Morph Step or Answer")
        return False

    return True


def load_and_preprocess_data(file_name):
    """Load and preprocess individual participant data."""
    try:
        data = pd.read_csv(os.path.join(FILES_ADDRESS, file_name), header=0)
        subject_id = int(file_name.split('_')[1])
        data['ID'] = subject_id

        if 'State' in data.columns:
            data = data[data['State'] == 1]

        cols_to_drop = ['State', 'Position ID', 'ITI',
                        'Trial_Onset', 'Stim_Onset', 'Stim_Offset']
        table = data.drop(
            columns=[c for c in cols_to_drop if c in data.columns])

        if validate_data(table):
            return table
        return pd.DataFrame()

    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        return pd.DataFrame()


def calculate_partial_residuals(table, model, variable, group_type):
    """Calculate partial residuals for the given variable."""
    if group_type == 'subject':
        unwanted_vars = ['Face_Gender', 'MS_FG', 'FG_SG']
        kept_vars = ['Morph Step', 'Subject_Gender', 'MS_SG']
    else:
        unwanted_vars = ['Subject_Gender', 'MS_SG', 'FG_SG']
        kept_vars = ['Morph Step', 'Face_Gender', 'MS_FG']

    unwanted_effects = sum(model.params[var] * table[var]
                           for var in unwanted_vars)
    residuals = table['Answer'] - unwanted_effects

    return residuals


def calculate_partial_r2(table, gender_label, group_type):
    """Calculate partial R² for each gender subgroup."""
    gender_data = table[table['Gender_Group'] == gender_label]
    x = gender_data['Morph Step']
    y = gender_data['Partial_Residuals']
    return np.corrcoef(x, y)[0, 1] ** 2


def create_subgroup_plot(table, model, group_type):
    """Create visualization for gender subgroup analysis with regression formulas."""
    if group_type == 'subject':
        group_col = 'Subject_Gender'
        palette = PALETTE_SUBJECT
        title_prefix = 'Particiapnt Gender'
    else:
        group_col = 'Face_Gender'
        palette = PALETTE_FACE
        title_prefix = 'Face Gender'

    table['Partial_Residuals'] = calculate_partial_residuals(
        table, model, 'Morph Step', group_type)
    table['Gender_Group'] = np.where(table[group_col] == 0, 'Female', 'Male')

    plt.figure(figsize=(9, 9))

    violin_data = table.copy()
    violin_data = violin_data.groupby(['Morph Step', 'Gender_Group'],
                                      group_keys=False).apply(
        lambda g: g[abs(g['Partial_Residuals'] - g['Partial_Residuals'].mean())
                    < 2 * g['Partial_Residuals'].std()]
    )

    sns.violinplot(data=violin_data, x='Morph Step', y='Partial_Residuals',
                   hue='Gender_Group', palette=palette, split=True,
                   inner=None, cut=3, gap=0.45, linewidth=1,
                   alpha=0.6, saturation=0.8, width=1.4)

    r2_values = {}
    formulas = []

    for gender_label, color in palette.items():
        gender_data = table[table['Gender_Group'] == gender_label]
        x = gender_data['Morph Step']
        y = gender_data['Partial_Residuals']
        slope, intercept = np.polyfit(x, y, 1)

        r2_values[gender_label] = calculate_partial_r2(
            table, gender_label, group_type)
        formula = f"{gender_label}: Answer = {intercept:.2f} + {slope:.2f} × MorphStep"
        formulas.append(formula)

        print()
        print(title_prefix)
        print(gender_label)
        print(formula)

        x_reg = np.array([1, 9])
        y_reg = slope * x_reg + intercept
        plt.plot(x_reg-1, y_reg, color=color, linestyle='--',
                 linewidth=2, alpha=0.8)

        means = gender_data.groupby('Morph Step')['Partial_Residuals'].mean()
        plt.scatter(means.index-1, means.values, color=color,
                    s=80, edgecolor='white', linewidth=1.5, zorder=3)

    title = (f"{title_prefix} Subgroups\n"
             f"Female R² = {r2_values['Female']:.3f}, "
             f"Male R² = {r2_values['Male']:.3f}")

    plt.title(title, pad=20, fontsize=11)
    plt.xlabel('Morphing Step | Others', fontsize=11)
    plt.ylabel('Particiapnts’ Rating (Partial Residual)', fontsize=11)
    plt.xticks(range(0, 9))
    plt.yticks(range(1, 10))
    plt.xlim(-1, 9)
    plt.ylim(0, 10)

    sns.despine()
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR,
                f'Validity_{title_prefix.replace(" ", "_")}_Subgroups.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()


def main():
    """Main analysis pipeline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_data = pd.DataFrame()
    for file_name in os.listdir(FILES_ADDRESS):
        if file_name.endswith('.csv'):
            table = load_and_preprocess_data(file_name)
            if not table.empty:
                all_data = pd.concat([all_data, table], ignore_index=True)

    if all_data.empty:
        print("Error: No valid data to analyze")
        return

    all_data['Face_Gender'] = np.where(all_data['Face Person'] < 12, 0, 1)
    gender_df = pd.read_csv(GENDER_FILE)
    all_data = all_data.merge(gender_df, on='ID')
    all_data['Subject_Gender'] = np.where(all_data['Gender'] == 'Female', 0, 1)

    all_data['MS_FG'] = all_data['Morph Step'] * all_data['Face_Gender']
    all_data['MS_SG'] = all_data['Morph Step'] * all_data['Subject_Gender']
    all_data['FG_SG'] = all_data['Face_Gender'] * all_data['Subject_Gender']

    predictors = ['Morph Step', 'Face_Gender', 'Subject_Gender',
                  'MS_FG', 'MS_SG', 'FG_SG']
    X = add_constant(all_data[predictors])
    model = sm.OLS(all_data['Answer'], X).fit()

    create_subgroup_plot(all_data, model, 'subject')
    create_subgroup_plot(all_data, model, 'face')

    with open(os.path.join(RESULTS_DIR, 'Model_Summary.txt'), 'w') as f:
        f.write(model.summary().as_text())


if __name__ == "__main__":
    main()
