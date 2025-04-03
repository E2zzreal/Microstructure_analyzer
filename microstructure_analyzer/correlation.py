import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential future warnings from pandas/numpy

def load_data(filepath, index_col=0):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    try:
        df = pd.read_csv(filepath, index_col=index_col)
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise

def merge_data(microstructure_features_df, external_data_df):
    """
    Merges microstructure features with external data based on their index (sample ID).

    Args:
        microstructure_features_df (pd.DataFrame): DataFrame containing microstructure features.
                                                   Index should be the sample ID.
        external_data_df (pd.DataFrame): DataFrame containing external data (elements, properties, etc.).
                                         Index should be the sample ID.

    Returns:
        pd.DataFrame: Merged DataFrame containing data only for common sample IDs.
                      Returns empty DataFrame if merge results in no common samples.
    """
    if microstructure_features_df.empty or external_data_df.empty:
        print("Warning: One or both input DataFrames for merging are empty.")
        return pd.DataFrame()

    # Perform an inner join to keep only common samples
    merged_df = pd.merge(microstructure_features_df, external_data_df,
                         left_index=True, right_index=True, how='inner')

    if merged_df.empty:
        print("Warning: No common samples found between feature data and external data.")

    return merged_df

def calculate_correlation(data_df, feature_columns, target_columns):
    """
    Calculates the correlation matrix between specified feature columns and target columns.

    Args:
        data_df (pd.DataFrame): Merged DataFrame containing both feature and target columns.
        feature_columns (list): List of column names representing the features.
        target_columns (list): List of column names representing the target properties.

    Returns:
        pd.DataFrame: Correlation matrix showing correlations between features and targets.
                      Returns empty DataFrame if input is invalid or calculation fails.
    """
    if data_df.empty or not feature_columns or not target_columns:
        print("Warning: Invalid input for correlation calculation.")
        return pd.DataFrame()

    # Check if all specified columns exist in the DataFrame
    missing_features = [col for col in feature_columns if col not in data_df.columns]
    missing_targets = [col for col in target_columns if col not in data_df.columns]

    if missing_features:
        print(f"Warning: Feature columns not found in DataFrame: {missing_features}")
        # Filter feature_columns to only include existing ones
        feature_columns = [col for col in feature_columns if col in data_df.columns]
    if missing_targets:
        print(f"Warning: Target columns not found in DataFrame: {missing_targets}")
        # Filter target_columns
        target_columns = [col for col in target_columns if col in data_df.columns]

    if not feature_columns or not target_columns:
        print("Warning: No valid feature or target columns remaining for correlation.")
        return pd.DataFrame()

    # Select relevant columns and calculate correlation
    try:
        correlation_matrix = data_df[feature_columns + target_columns].corr()
        # Extract the part of the matrix showing feature vs target correlations
        feature_target_corr = correlation_matrix.loc[feature_columns, target_columns]
        return feature_target_corr
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return pd.DataFrame()


# Example usage (if run as a script)
if __name__ == '__main__':
    # Example paths from notebook - ADJUST AS NEEDED
    FILTERED_FEATURES_PATH = 'filtered_features_v2.csv' # Assumes this file exists from analysis step
    EXTERNAL_DATA_PATH = '/home/kemove/Desktop/Mag/2-ZH/database-250211.csv' # Example path from notebook
    # Define column groups based on notebook analysis
    # Note: 'area_mean' was added separately in the notebook, need to ensure it's in the filtered features file
    # or load the unfiltered features and select it. Assuming it's in FILTERED_FEATURES_PATH for now.
    PROPS_COLS = ['Br20','Hc20','Hc147','Hr20','Hr147']
    PPS_COLS = ['C','F','J','L']
    EFFS_COLS = ['Neff','alpha']
    PCS_COLS = ['PC5-X10000A']
    # ELES_COLS = ['B', ..., 'Fe'] # Define element columns if needed for correlation

    print("Running correlation module example...")
    print(f"Filtered Features Path: {FILTERED_FEATURES_PATH}")
    print(f"External Data Path: {EXTERNAL_DATA_PATH}")

    if not os.path.exists(FILTERED_FEATURES_PATH):
        print(f"Error: Filtered features file '{FILTERED_FEATURES_PATH}' not found. Cannot run example.")
    elif not os.path.exists(EXTERNAL_DATA_PATH):
        print(f"Error: External data file '{EXTERNAL_DATA_PATH}' not found. Cannot run example.")
    else:
        try:
            # Load data
            features_df = load_data(FILTERED_FEATURES_PATH, index_col=0)
            external_df = load_data(EXTERNAL_DATA_PATH, index_col=0)

            # Define microstructure feature columns to use (all columns from features_df)
            # Ensure 'area_mean' or its equivalent global feature name is included if needed.
            # Example: Check if 'global_area_mean' exists from feature_extraction output
            ms_features = features_df.columns.tolist()
            if 'global_area_mean' in ms_features: # Check for the prefixed version
                 ms_features_with_area = ms_features
            elif 'area_mean' in ms_features: # Check for non-prefixed version
                 ms_features_with_area = ms_features
            else:
                 print("Warning: 'area_mean' or 'global_area_mean' not found in feature columns. Correlation might differ from notebook.")
                 ms_features_with_area = ms_features # Proceed without it if not found


            # Merge data
            merged_data = merge_data(features_df, external_df)

            if not merged_data.empty:
                print(f"\nMerged data shape: {merged_data.shape}")

                # Calculate correlations for different target groups
                print("\nCorrelation: Microstructure Features vs. Magnetic Properties (props)")
                corr_props = calculate_correlation(merged_data, ms_features_with_area, PROPS_COLS)
                # Sort by a specific property as in notebook (e.g., Hr147)
                if 'Hr147' in corr_props.columns:
                     print(corr_props.sort_values(by='Hr147', ascending=False))
                else:
                     print(corr_props)


                print("\nCorrelation: Microstructure Features vs. Process Parameters (pps)")
                corr_pps = calculate_correlation(merged_data, ms_features_with_area, PPS_COLS)
                if 'J' in corr_pps.columns: # Example sort
                     print(corr_pps.sort_values(by='J', ascending=False))
                else:
                     print(corr_pps)

                print("\nCorrelation: Microstructure Features vs. Efficiency (effs)")
                corr_effs = calculate_correlation(merged_data, ms_features_with_area, EFFS_COLS)
                if 'Neff' in corr_effs.columns: # Example sort
                     print(corr_effs.sort_values(by='Neff', ascending=False))
                else:
                     print(corr_effs)

                print("\nCorrelation: Microstructure Features vs. PC (pcs)")
                corr_pcs = calculate_correlation(merged_data, ms_features_with_area, PCS_COLS)
                if 'PC5-X10000A' in corr_pcs.columns: # Example sort
                     print(corr_pcs.sort_values(by='PC5-X10000A', ascending=False))
                else:
                     print(corr_pcs)

            else:
                print("Merged data is empty, skipping correlation calculations.")

        except Exception as main_e:
            print(f"An error occurred during the example run: {main_e}")

    print("\nCorrelation module example finished.")
