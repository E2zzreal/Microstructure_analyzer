import numpy as np
import pandas as pd
import warnings
from scipy.stats import f_oneway, shapiro, levene
from sklearn.feature_selection import VarianceThreshold

# Try importing pingouin, handle if not installed
try:
    import pingouin as pg
except ImportError:
    pg = None
    warnings.warn("Pingouin library not found. ICC calculation will not be available.")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_analysis_dataframe(features_df):
    """
    Prepares the feature DataFrame for consistency analysis by adding
    'Sample' and 'Rater' columns.

    Args:
        features_df (pd.DataFrame): DataFrame where the index represents sample IDs
                                    (potentially duplicated for multiple measurements
                                    per sample).

    Returns:
        pd.DataFrame: DataFrame with 'Sample' (original index) and 'Rater'
                      (within-sample measurement number) columns added as the
                      first two columns. Returns empty DataFrame if input is empty.
    """
    if features_df.empty:
        return pd.DataFrame()

    df_analysis = features_df.copy()
    # Use the existing index as the 'Sample' identifier
    df_analysis.insert(0, 'Sample', df_analysis.index)
    # Calculate 'Rater' based on cumulative count within each sample group
    df_analysis.insert(1, 'Rater', df_analysis.groupby('Sample').cumcount() + 1)
    # Reset index if needed, or keep Sample as index? Notebook kept Sample as index.
    # df_analysis = df_analysis.reset_index(drop=True) # Optional: remove original index

    # Attempt to convert all feature columns to numeric, coercing errors
    feature_cols = df_analysis.columns[2:] # Skip Sample, Rater
    for col in feature_cols:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

    # Check for columns that could not be converted (all NaN)
    cols_to_drop = [col for col in feature_cols if df_analysis[col].isnull().all()]
    if cols_to_drop:
        print(f"Warning: Dropping columns with all non-numeric values: {cols_to_drop}")
        df_analysis = df_analysis.drop(columns=cols_to_drop)

    # Handle remaining NaNs if necessary (e.g., fill with 0 or mean)
    # For now, let downstream functions handle NaNs (like pingouin's nan_policy)
    # df_analysis = df_analysis.fillna(0) # Example: fill NaNs with 0

    return df_analysis


def calculate_icc(data, features, icc_threshold=0.75, max_raters=8):
    """
    Calculates Intraclass Correlation Coefficient (ICC) for specified features.
    Uses the ICC(3,k) model (Two-way random effects, consistency, average rater).

    Args:
        data (pd.DataFrame): DataFrame prepared by `prepare_analysis_dataframe`,
                             containing 'Sample', 'Rater', and feature columns.
        features (list): List of feature column names to calculate ICC for.
        icc_threshold (float): The minimum ICC value for a feature to be considered reliable.
        max_raters (int): Maximum number of raters (measurements per sample) to include
                          in the ICC calculation (as done in the notebook).

    Returns:
        pd.DataFrame: DataFrame with 'Feature' and 'ICC' columns. Returns empty
                      DataFrame if pingouin is not installed or data is unsuitable.
    """
    if pg is None:
        print("Error: Pingouin library is required for ICC calculation.")
        return pd.DataFrame(columns=['Feature', 'ICC'])

    if 'Sample' not in data.columns or 'Rater' not in data.columns:
        print("Error: Data must contain 'Sample' and 'Rater' columns for ICC.")
        return pd.DataFrame(columns=['Feature', 'ICC'])

    icc_results = []
    # Filter data based on max_raters
    data_filtered = data[data['Rater'] <= max_raters].copy()

    if data_filtered.empty:
         print(f"Warning: No data available for ICC calculation with max_raters <= {max_raters}.")
         return pd.DataFrame(columns=['Feature', 'ICC'])


    for feat in features:
        if feat not in data_filtered.columns:
            print(f"Warning: Feature '{feat}' not found in data for ICC calculation. Skipping.")
            continue

        # Prepare data subset for pingouin
        icc_data_subset = data_filtered[['Sample', 'Rater', feat]].dropna() # Drop rows with NaN in this feature

        # Check if enough data remains after dropping NaNs
        if icc_data_subset['Sample'].nunique() < 2 or icc_data_subset.empty:
             print(f"Warning: Not enough data or samples for feature '{feat}' after dropping NaNs. Skipping ICC.")
             icc_value = np.nan # Assign NaN if calculation cannot be performed
        else:
            try:
                # Calculate ICC using pingouin
                icc = pg.intraclass_corr(data=icc_data_subset,
                                         targets='Sample',
                                         raters='Rater',
                                         ratings=feat,
                                         nan_policy='omit') # Should be handled by dropna already

                # Extract ICC(3,k) value
                # Use .loc with error handling in case 'ICC3k' is not present
                icc_value = icc.set_index('Type').loc['ICC3k', 'ICC'] if 'ICC3k' in icc['Type'].values else np.nan

            except Exception as e:
                print(f"Error calculating ICC for feature '{feat}': {e}")
                icc_value = np.nan # Assign NaN on error

        icc_results.append({'Feature': feat, 'ICC': icc_value})

    icc_df = pd.DataFrame(icc_results)
    # Filter based on threshold (optional here, can be done later)
    # reliable_features = icc_df[icc_df['ICC'] >= icc_threshold]['Feature'].tolist()
    return icc_df


def perform_anova(data, features, alpha=0.05):
    """
    Performs one-way ANOVA for specified features to test differences between Samples.
    Applies Bonferroni correction for multiple comparisons.

    Args:
        data (pd.DataFrame): DataFrame prepared by `prepare_analysis_dataframe`.
        features (list): List of feature column names to perform ANOVA on.
        alpha (float): Significance level before correction.

    Returns:
        pd.DataFrame: DataFrame with 'Feature', 'P_value', and 'Reject_H0' (after correction).
                      Returns empty DataFrame if data is unsuitable.
    """
    if 'Sample' not in data.columns:
        print("Error: Data must contain 'Sample' column for ANOVA.")
        return pd.DataFrame(columns=['Feature', 'P_value', 'Reject_H0'])

    anova_results = []
    num_features = len(features)
    if num_features == 0:
        return pd.DataFrame(columns=['Feature', 'P_value', 'Reject_H0'])

    # Bonferroni corrected alpha
    corrected_alpha = alpha / num_features

    unique_samples = data['Sample'].unique()
    if len(unique_samples) < 2:
        print("Warning: ANOVA requires at least two samples. Skipping.")
        return pd.DataFrame(columns=['Feature', 'P_value', 'Reject_H0'])


    for feat in features:
        if feat not in data.columns:
            print(f"Warning: Feature '{feat}' not found in data for ANOVA calculation. Skipping.")
            continue

        # Prepare groups for ANOVA, handling potential NaNs within groups
        groups = [data[data['Sample'] == sample][feat].dropna().values for sample in unique_samples]
        # Filter out empty groups or groups with insufficient data for variance calculation
        valid_groups = [g for g in groups if len(g) >= 2] # Need at least 2 points per group for variance

        p_value = np.nan # Default if ANOVA cannot be run
        reject_h0 = False # Default

        if len(valid_groups) < 2: # Need at least two valid groups for comparison
            print(f"Warning: Not enough valid groups (>=2 data points) for feature '{feat}'. Skipping ANOVA.")
        else:
            try:
                # Check for zero variance within all groups (can cause issues)
                variances = [np.var(g) for g in valid_groups]
                if np.all(np.isclose(variances, 0)):
                     print(f"Warning: All groups have zero variance for feature '{feat}'. Skipping ANOVA.")
                     p_value = 1.0 # Or NaN? Assign 1.0 as no difference detected.
                else:
                    f_stat, p_value = f_oneway(*valid_groups)
                    reject_h0 = p_value < corrected_alpha

            except Exception as e:
                print(f"Error performing ANOVA for feature '{feat}': {e}")
                p_value = np.nan # Assign NaN on error

        anova_results.append({
            'Feature': feat,
            'P_value': p_value,
            'Reject_H0': reject_h0
        })

    return pd.DataFrame(anova_results)


def perform_assumption_checks(data, features):
    """
    Performs normality (Shapiro-Wilk) and homogeneity of variance (Levene) tests.
    Note: These are exploratory checks; ANOVA is somewhat robust to violations.

    Args:
        data (pd.DataFrame): DataFrame prepared by `prepare_analysis_dataframe`.
        features (list): List of feature column names to check.

    Returns:
        dict: A dictionary containing results for 'normality' and 'homogeneity'.
              Normality results map feature -> sample -> p-value.
              Homogeneity results map feature -> p-value.
    """
    normality_results = defaultdict(dict)
    homogeneity_results = {}

    if 'Sample' not in data.columns:
        print("Error: Data must contain 'Sample' column for assumption checks.")
        return {'normality': {}, 'homogeneity': {}}

    unique_samples = data['Sample'].unique()

    for feat in features:
        if feat not in data.columns:
            print(f"Warning: Feature '{feat}' not found in data for assumption checks. Skipping.")
            continue

        # Normality Check (Shapiro-Wilk per sample)
        for sample in unique_samples:
            sample_data = data[data['Sample'] == sample][feat].dropna()
            p_val_shapiro = np.nan
            # Shapiro test requires at least 3 data points
            if len(sample_data) >= 3:
                try:
                    stat, p_val_shapiro = shapiro(sample_data)
                except Exception as e:
                    # print(f"Warning: Shapiro test failed for {feat}, sample {sample}: {e}")
                    p_val_shapiro = np.nan
            normality_results[feat][sample] = p_val_shapiro

        # Homogeneity of Variance Check (Levene test across samples)
        groups = [data[data['Sample'] == sample][feat].dropna().values for sample in unique_samples]
        valid_groups = [g for g in groups if len(g) > 0] # Levene needs non-empty groups
        p_val_levene = np.nan
        if len(valid_groups) >= 2: # Need at least two groups
             try:
                  stat, p_val_levene = levene(*valid_groups)
             except Exception as e:
                  # print(f"Warning: Levene test failed for {feat}: {e}")
                  p_val_levene = np.nan
        homogeneity_results[feat] = p_val_levene

    return {'normality': dict(normality_results), 'homogeneity': homogeneity_results}


def filter_features_icc_anova(icc_df, anova_df, icc_threshold=0.75):
    """
    Filters features based on combined ICC and ANOVA results.

    Args:
        icc_df (pd.DataFrame): DataFrame from calculate_icc().
        anova_df (pd.DataFrame): DataFrame from perform_anova().
        icc_threshold (float): Minimum acceptable ICC value.

    Returns:
        pd.DataFrame: Merged DataFrame with an added 'Good_Feature' boolean column.
    """
    if icc_df.empty or anova_df.empty:
        print("Warning: Cannot filter features due to empty ICC or ANOVA results.")
        return pd.DataFrame()

    # Merge results
    results = pd.merge(icc_df, anova_df, on='Feature', how='inner')

    # Apply filtering criteria
    # Ensure Reject_H0 is boolean, handle potential NaNs in ICC or P_value
    results['Reject_H0'] = results['Reject_H0'].fillna(False).astype(bool)
    results['ICC'] = pd.to_numeric(results['ICC'], errors='coerce') # Ensure numeric, coerce errors

    results['Good_Feature'] = (results['ICC'] >= icc_threshold) & (results['Reject_H0'])

    return results


def filter_features_variance(features_df, threshold=0.0):
    """
    Filters features based on variance threshold.

    Args:
        features_df (pd.DataFrame): DataFrame with features (samples as rows, features as columns).
                                    Should contain only numeric features.
        threshold (float): Variance threshold. Features with variance below this will be removed.

    Returns:
        list: List of feature names that meet the variance threshold.
    """
    if features_df.empty:
        return []

    # Ensure data is numeric, handle potential errors
    numeric_df = features_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    if numeric_df.empty:
         print("Warning: No numeric features found for variance filtering.")
         return []

    # Fill remaining NaNs if any (e.g., with mean or 0) before variance calculation
    numeric_df = numeric_df.fillna(numeric_df.mean()) # Example: fill with mean

    selector = VarianceThreshold(threshold=threshold)
    try:
        selector.fit(numeric_df)
        selected_features = numeric_df.columns[selector.get_support()].tolist()
        removed_features = numeric_df.columns[~selector.get_support()].tolist()
        if removed_features:
             print(f"Variance Threshold Filter: Removed {len(removed_features)} features: {removed_features}")
        return selected_features
    except Exception as e:
        print(f"Error during variance threshold filtering: {e}")
        return numeric_df.columns.tolist() # Return all columns on error


def filter_features_correlation(features_df, threshold=0.95):
    """
    Iteratively filters features based on high correlation.
    In each step, identifies the feature with the most highly correlated pairs
    (> threshold) and removes it. Prioritizes removing features ending in '_std'.

    Args:
        features_df (pd.DataFrame): DataFrame with features (samples as rows, features as columns).
                                    Should contain only numeric features.
        threshold (float): Correlation threshold. Pairs above this are considered highly correlated.

    Returns:
        list: List of feature names remaining after correlation filtering.
    """
    if features_df.empty:
        return []

    # Ensure data is numeric
    numeric_df = features_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if numeric_df.empty:
         print("Warning: No numeric features found for correlation filtering.")
         return []
    # Fill NaNs before correlation calculation
    numeric_df = numeric_df.fillna(numeric_df.mean())

    feature_list = numeric_df.columns.tolist()
    removed_count = 0

    while True:
        if len(feature_list) < 2:
            break # Need at least 2 features to correlate

        corr_matrix = numeric_df[feature_list].corr().abs()
        # Create mask for upper triangle (excluding diagonal) where correlation > threshold
        upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_mask = (corr_matrix > threshold) & upper_tri_mask

        # Count high correlations per feature
        corr_counts = high_corr_mask.sum(axis=0) + high_corr_mask.sum(axis=1) # Sum rows and columns

        if corr_counts.max() == 0:
            break # No more high correlations found

        # Find feature(s) with the maximum number of high correlations
        max_count = corr_counts.max()
        candidates = corr_counts[corr_counts == max_count].index.tolist()

        # Prioritize removing features ending in '_std' among candidates
        std_candidates = [feat for feat in candidates if feat.endswith('_std')]

        if std_candidates:
            remove_feature = std_candidates[0] # Remove the first _std candidate
        else:
            remove_feature = candidates[0] # Remove the first candidate if no _std found

        feature_list.remove(remove_feature)
        removed_count += 1
        print(f"Correlation Filter: Removing '{remove_feature}' (highly correlated with {max_count} other features).")

    print(f"Correlation Filter: Removed {removed_count} features in total.")
    return feature_list


# Example usage (if run as a script)
if __name__ == '__main__':
    # Create a dummy DataFrame similar to the notebook's structure
    data = {
        'Sample': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Rater': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Feature1': [1.1, 1.0, 1.2, 5.0, 5.1, 4.9, 3.0, 3.1, 2.9], # Good ICC, Good ANOVA
        'Feature2': [10, 20, 15, 12, 22, 18, 11, 21, 16],          # Poor ICC, Good ANOVA
        'Feature3': [100, 101, 100, 102, 100, 101, 101, 100, 102], # Good ICC, Poor ANOVA
        'Feature4': [5, 5, 5, 5, 5, 5, 5, 5, 5],                   # Zero variance
        'Feature5': [1.1, 1.0, 1.2, 5.0, 5.1, 4.9, 3.0, 3.1, 2.9], # Duplicate of F1
        'Feature5_std': [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], # Correlated with F5
    }
    dummy_df = pd.DataFrame(data)
    all_features = dummy_df.columns[2:].tolist()

    print("Running analysis module example...")

    # 1. ICC Calculation
    icc_results_df = calculate_icc(dummy_df, all_features, max_raters=3)
    print("\nICC Results:")
    print(icc_results_df)

    # 2. ANOVA Calculation
    anova_results_df = perform_anova(dummy_df, all_features, alpha=0.05)
    print("\nANOVA Results:")
    print(anova_results_df)

    # 3. Assumption Checks (Optional)
    # assumption_results = perform_assumption_checks(dummy_df, all_features)
    # print("\nAssumption Checks (P-values):")
    # print(" Normality:", assumption_results['normality'])
    # print(" Homogeneity:", assumption_results['homogeneity'])

    # 4. Filter based on ICC and ANOVA
    combined_results = filter_features_icc_anova(icc_results_df, anova_results_df, icc_threshold=0.75)
    print("\nCombined ICC/ANOVA Filter Results:")
    print(combined_results)
    good_features_icc_anova = combined_results[combined_results['Good_Feature']]['Feature'].tolist()
    print(f"\nFeatures passing ICC & ANOVA: {good_features_icc_anova}")

    # Prepare DataFrame for variance/correlation filtering (using mean per sample)
    # In a real scenario, this would come from batch_calculate_features output
    features_for_filtering = dummy_df.groupby('Sample')[all_features].mean()
    print("\nMean Features per Sample (for Variance/Correlation):")
    print(features_for_filtering)


    # 5. Variance Threshold Filter
    features_passing_variance = filter_features_variance(features_for_filtering, threshold=0.0)
    print(f"\nFeatures passing Variance Threshold: {features_passing_variance}")

    # 6. Correlation Filter
    final_features = filter_features_correlation(features_for_filtering[features_passing_variance], threshold=0.95)
    print(f"\nFinal Features after Correlation Filter: {final_features}")

    print("\nAnalysis module example finished.")
