import statsmodels.api as sm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
Validity Analysis for STEMorph

This script performs validity analysis on STEMorph, a set of morphed
emotional face stimuli ranging from angry to happy. It processes participant ratings,
performs regression analysis with gender effects and interactions, and generates 
visualizations.

Author: Mohammad Ebrahim Katebi
"""

# Constants
FILES_ADDRESS = '../Data_Validity/'
RESULTS_DIR = '../Results_Validity/'
GENDER_FILE = '../Subject_Gender.csv'


def validate_data(table):
    """
    Validate the data table has all required columns and valid values.
    """
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
    """
    Load and preprocess data from a CSV file.
    """
    try:
        data = pd.read_csv(os.path.join(FILES_ADDRESS, file_name), header=0)
        subject_id = int(file_name.split('_')[1])
        data['ID'] = subject_id

        if 'State' in data.columns:
            data = data[data['State'] == 1]

        columns_to_drop = ['State', 'Position ID', 'ITI',
                           'Trial_Onset', 'Stim_Onset', 'Stim_Offset']
        columns_to_drop = [
            col for col in columns_to_drop if col in data.columns]
        table = data.drop(columns=columns_to_drop)

        if validate_data(table):
            return table
        return pd.DataFrame()

    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        return pd.DataFrame()


def create_interaction_terms(table):
    """
    Create all possible interaction terms for the regression.
    """
    table['MS_FG'] = table['Morph Step'] * table['Face_Gender']
    table['MS_SG'] = table['Morph Step'] * table['Subject_Gender']
    table['FG_SG'] = table['Face_Gender'] * table['Subject_Gender']
    return table


def calculate_partial_residuals(table, model, variable):
    """
    Calculate partial residuals.
    """
    unwanted_vars = ['Subject_Gender',
                     'Face_Gender', 'MS_SG', 'MS_FG', 'FG_SG']

    unwanted_effects = sum(model.params[var] * table[var]
                           for var in unwanted_vars)
    residuals = table['Answer'] - unwanted_effects

    return residuals


def remove_outliers(table):
    """
    Remove outliers from the data based on 2 standard deviations.
    """
    cleaned_table = table.copy()
    for morph_step in np.unique(table['Morph Step']):
        step_data = table[table['Morph Step'] == morph_step]
        mean = step_data['Partial Residuals'].mean()
        std = step_data['Partial Residuals'].std()
        cleaned_table = cleaned_table[
            (cleaned_table['Morph Step'] != morph_step) |
            ((cleaned_table['Morph Step'] == morph_step) &
             (abs(cleaned_table['Partial Residuals'] - mean) < 2 * std))
        ]
    return cleaned_table


def create_coefficient_plot(model, file_name):
    """
    Create and save coefficient plot showing beta estimates and CIs.
    """
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    coefs = pd.DataFrame({
        'term': model.params.index,
        'estimate': model.params.values
    })
    cis = model.conf_int()
    cis.columns = ['ci_low', 'ci_high']

    coef_df = coefs.copy()
    coef_df['ci_low'] = cis['ci_low'].values
    coef_df['ci_high'] = cis['ci_high'].values
    coef_df = coef_df[coef_df['term'] != 'const']
    coef_df['pvalue'] = model.pvalues[1:].values

    y_pos = np.arange(len(coef_df))
    colors = plt.cm.Dark2(np.linspace(0, 1, len(coef_df)))

    ax.barh(y_pos, coef_df['estimate'],
            xerr=np.abs(coef_df[['ci_low', 'ci_high']].values -
                        coef_df['estimate'].values.reshape(-1, 1)).T,
            color=colors, height=0.7, capsize=3)

    for i, p_val in enumerate(coef_df['pvalue']):
        if p_val < 0.05:
            if coef_df['estimate'].iloc[i] > 0:
                ax.text(coef_df['ci_high'].iloc[i] + 0.08, i - 0.1, '*',
                        ha='center', va='center', fontsize=18)
            if coef_df['estimate'].iloc[i] <= 0:
                ax.text(coef_df['ci_high'].iloc[i] - 0.18, i - 0.1, '*',
                        ha='center', va='center', fontsize=18)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['term'])
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('β Estimate (95% CI)')
    ax.set_title('Regression Coefficients')

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{file_name} Coefficients.png'),
                dpi=400, bbox_inches='tight', pad_inches=0.25)
    plt.close()


def create_validity_plot(table, file_name):
    """
    Create and save validity plot with regression analysis.
    """
    try:
        # Data preparation
        table['Face_Gender'] = np.where(table['Face Person'] < 12, 0, 1)
        gender_df = pd.read_csv(GENDER_FILE)
        table = table.merge(gender_df, left_on='ID',
                            right_on='ID', how='inner')
        table['Subject_Gender'] = np.where(table['Gender'] == 'Female', 0, 1)
        table = create_interaction_terms(table)

        # Model fitting
        predictors = ['Morph Step', 'Face_Gender', 'Subject_Gender',
                      'MS_FG', 'MS_SG', 'FG_SG']
        X = sm.add_constant(table[predictors])
        y = table['Answer']
        model = sm.OLS(y, X).fit()
        partial_resid = calculate_partial_residuals(table, model, 'Morph Step')

        # Plot setup
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        aspect_ratio = bbox.width / bbox.height

        # Color palette
        palette = sns.color_palette("vlag", np.unique(table['Answer']).size)
        palette = [sns.desaturate(color, 0.8) for color in palette]
        palette.reverse()

        # Scatter plot with jitter
        for step in range(1, 10):
            mask = table['Morph Step'] == step
            step_resid = partial_resid[mask]
            n_points = len(step_resid)

            radius = np.random.uniform(0, 0.1, n_points)
            theta = np.random.uniform(0, 2 * np.pi, n_points)
            x_jitter = (step - 1) + radius * np.cos(theta)
            y_jitter = step_resid + radius * np.sin(theta) * aspect_ratio

            plt.scatter(x_jitter, y_jitter, color='black',
                        alpha=0.2, s=1, zorder=1)

        # Violin plots
        violin_data = pd.DataFrame({
            'Morph Step': table['Morph Step'],
            'Partial Residuals': partial_resid
        })
        violin_data_clean = remove_outliers(violin_data)

        sns.violinplot(data=violin_data_clean, x='Morph Step',
                       y='Partial Residuals', palette=palette,
                       inner=None, cut=3, width=0.7,
                       linewidth=1, alpha=0.5, zorder=2)

        # Mean points and regression line
        means = violin_data.groupby('Morph Step')['Partial Residuals'].mean()
        plt.scatter(means.index - 1, means.values,
                    color='black', s=50, zorder=4)

        x_reg = np.array([0, 8])
        slope = model.params['Morph Step']
        intercept = partial_resid.mean() - slope * \
            (table['Morph Step'].mean() - 1)
        y_reg = slope * x_reg + intercept
        plt.plot(x_reg, y_reg, 'k:', linewidth=1, zorder=3)

        # Plot formatting
        plt.xticks(range(0, 9))
        plt.yticks(range(1, 10))
        plt.xlim(-1, 9)
        plt.ylim(0, 10)

        # Title and labels with formula
        formula = (
            f"Answer = {model.params['Morph Step']:.3f} × MS + "
            f"{model.params['Face_Gender']:.3f} × FG + "
            f"{model.params['Subject_Gender']:.3f} × SG + "
            f"{model.params['MS_FG']:.3f} × (MS×FG) + "
            f"{model.params['MS_SG']:.3f} × (MS×SG) + "
            f"{model.params['FG_SG']:.3f} × (FG×SG) + "
            f"{model.params['const']:.3f}"
        )

        plt.title(
            f"Validity of Emotional Ratings Across Morphing Steps\nR2 = {model.rsquared:.3f}", pad=25, fontsize=16)
        plt.xlabel('Morphing Step | Others', fontsize=14)
        plt.ylabel('Participants\' Rating(Partial Residual)', fontsize=14)

        sns.despine()
        plt.tight_layout()

        # Save outputs
        plt.savefig(os.path.join(RESULTS_DIR, f'{file_name} Regression.png'),
                    dpi=400, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        create_coefficient_plot(model, file_name)

        # Save model statistics
        with open(os.path.join(RESULTS_DIR, f'{file_name}_model_stats.txt'), 'w') as f:
            f.write(model.summary().as_text())

        # Save processed data
        final_table = pd.DataFrame({
            'ID': table['ID'],
            'Face_Person': table['Face Person'],
            'Morph_Step': table['Morph Step'],
            'Answer': table['Answer'],
            'Face_Gender': table['Face_Gender'],
            'Subject_Gender': table['Subject_Gender'],
            'MS_FG': table['MS_FG'],
            'MS_SG': table['MS_SG'],
            'FG_SG': table['FG_SG'],
            'Partial_Residuals': partial_resid,
            'Predicted_Values': model.predict(X)
        })

        final_table.to_csv(os.path.join(RESULTS_DIR, f'{file_name}_processed_data.csv'),
                           index=False)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

    return model


def main():
    """
    Main function to run the validity analysis.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    list_of_files = [f for f in os.listdir(
        FILES_ADDRESS) if f.endswith('.csv')]
    if not list_of_files:
        print("Error: No CSV files found in data directory")
        return

    table_all = pd.DataFrame()

    for file_name in list_of_files:
        try:
            table = load_and_preprocess_data(file_name)
            if not table.empty:
                table_all = pd.concat([table_all, table], axis=0)
                print(f"Processed {file_name}")
            else:
                print(f"Warning: Empty data in {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    if table_all.empty:
        print("Error: No valid data to analyze")
        return

    create_validity_plot(table_all, 'Validity - Subject Average')


if __name__ == "__main__":
    main()
