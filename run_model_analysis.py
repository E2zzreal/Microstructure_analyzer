import argparse
import os
import sys
import time
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
DEFAULT_TARGET_COL = 'Hc20'
DEFAULT_OUTPUT_DIR = 'results/model_analysis'
DEFAULT_CV_FOLDS = 5
DEFAULT_RFE_STEP = 1
DEFAULT_RANDOM_STATE = 42

# --- Helper Functions ---

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Add seeds for other libraries if needed (e.g., torch, tensorflow)

def load_and_prepare_data(feature_csv, external_csv, target_col):
    """Loads features and external data, merges, preprocesses, and splits."""
    print(f"Loading features from: {feature_csv}")
    if not os.path.exists(feature_csv):
        print(f"Error: Feature file not found: {feature_csv}")
        sys.exit(1)
    features_df = pd.read_csv(feature_csv, index_col=0)

    print(f"Loading external data from: {external_csv}")
    if not os.path.exists(external_csv):
        print(f"Error: External data file not found: {external_csv}")
        sys.exit(1)
    external_df = pd.read_csv(external_csv, index_col=0)

    # Merge data
    print("Merging feature data and external data...")
    merged_df = pd.merge(features_df, external_df, left_index=True, right_index=True, how='inner')

    if merged_df.empty:
        print("Error: No common samples found after merging. Exiting.")
        sys.exit(1)
    print(f"Merged data shape: {merged_df.shape}")

    # Separate features (X) and target (y)
    if target_col not in merged_df.columns:
        print(f"Error: Target column '{target_col}' not found in merged data.")
        print(f"Available columns: {merged_df.columns.tolist()}")
        sys.exit(1)

    y = merged_df[target_col]
    X = merged_df.drop(columns=[target_col] + list(external_df.columns.difference([target_col]))) # Keep only microstructure features

    # Ensure all feature columns are numeric, coerce errors and check
    original_feature_names = X.columns.tolist()
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().any().any():
        print("Warning: Non-numeric values found in feature columns after coercion. Imputing NaNs.")

    # Preprocessing Pipeline (Imputation + Scaling)
    # Using Pipeline to prevent data leakage during CV
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # Impute missing values with mean
        ('scaler', StandardScaler()) # Scale features
    ])

    print("Applying preprocessing (imputation and scaling)...")
    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed, columns=original_feature_names, index=X.index) # Keep index for potential later use

    print(f"Target variable: '{target_col}'")
    print(f"Features shape after preprocessing: {X_processed_df.shape}")

    return X_processed_df, y, original_feature_names, preprocessor

def perform_rfecv(X, y, cv_folds, rfe_step, random_state):
    """Performs Recursive Feature Elimination with Cross-Validation."""
    print("\n--- Starting RFECV ---")
    estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # Note: RFECV scoring should use metrics where higher is better (like R2)
    # or negative versions of error metrics (neg_mean_squared_error)
    selector = RFECV(estimator=estimator, step=rfe_step, cv=cv,
                     scoring='r2', # Use R2 score for selection
                     n_jobs=-1,    # Use all available cores
                     verbose=1)     # Show progress

    start_time = time.time()
    selector.fit(X.values, y.values) # RFECV expects numpy arrays
    end_time = time.time()
    print(f"RFECV completed in {end_time - start_time:.2f} seconds.")

    selected_mask = selector.support_
    optimal_n_features = selector.n_features_
    selected_features = X.columns[selected_mask].tolist()

    print(f"Optimal number of features selected by RFECV: {optimal_n_features}")
    print(f"Selected features: {selected_features}")

    # Plot RFECV results (optional but helpful)
    try:
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (R^2)")
        # Check if grid_scores_ attribute exists (newer sklearn versions use cv_results_)
        if hasattr(selector, 'grid_scores_'):
             scores = selector.grid_scores_
        elif hasattr(selector, 'cv_results_') and 'mean_test_score' in selector.cv_results_:
             scores = selector.cv_results_['mean_test_score']
        else:
             scores = None
             print("Warning: Could not retrieve RFECV scores for plotting.")

        if scores is not None:
             plt.plot(range(1, len(scores) + 1), scores)
             plt.title('RFECV Performance vs. Number of Features')
             plt.grid(True)
             # Mark the optimal point
             plt.scatter(optimal_n_features, scores[optimal_n_features-1], s=100,
                         facecolors='none', edgecolors='r', label=f'Optimal ({optimal_n_features} features)')
             plt.legend()
             # Save the plot later in the main function
             # plt.savefig(os.path.join(output_dir, 'rfecv_performance.png'))
             # plt.close()
    except Exception as e:
        print(f"Warning: Could not plot RFECV results: {e}")


    return selected_mask, selected_features, selector

def train_and_evaluate_models(X_rfe, y, selected_feature_names, cv_folds, random_state, output_dir):
    """Trains, tunes, and evaluates multiple models using GridSearchCV."""
    print("\n--- Starting Model Training and Evaluation (using selected features) ---")
    models = {
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=1), # n_jobs=1 for XGBoost within GridSearchCV
        'SVR': SVR(),
        'Lasso': Lasso(random_state=random_state),
        'Ridge': Ridge(random_state=random_state),
        'GaussianProcess': GaussianProcessRegressor(random_state=random_state, normalize_y=True) # Normalize target for GPR stability
    }

    # Define parameter grids (adjust ranges as needed)
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        'Lasso': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'Ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'GaussianProcess': {
            # Define different kernels to try
            'kernel': [C(1.0) * RBF(1.0), C(1.0) * Matern(length_scale=1.0, nu=1.5)]
            # GPR hyperparameter tuning is often done differently (optimizing marginal likelihood)
            # GridSearchCV might not be the best approach here, but we include it for consistency.
            # Consider fitting GPR separately if more control over kernel optimization is needed.
        }
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}
    best_overall_score = -np.inf
    best_overall_model_name = None
    best_overall_estimator = None

    X_rfe_values = X_rfe.values # Use numpy array for sklearn functions
    y_values = y.values

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        param_grid = param_grids[name]

        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv,
                                       scoring='r2', n_jobs=-1, verbose=1)
            grid_search.fit(X_rfe_values, y_values)

            best_estimator = grid_search.best_estimator_
            best_score = grid_search.best_score_ # Mean R2 score from CV
            best_params = grid_search.best_params_

            # Calculate additional metrics using cross_val_predict
            y_pred_cv = cross_val_predict(best_estimator, X_rfe_values, y_values, cv=cv, n_jobs=-1)
            mae_cv = mean_absolute_error(y_values, y_pred_cv)
            rmse_cv = np.sqrt(mean_squared_error(y_values, y_pred_cv))

            end_time = time.time()
            print(f"{name} finished in {end_time - start_time:.2f} seconds.")
            print(f"  Best CV R2: {best_score:.4f}")
            print(f"  Best Params: {best_params}")
            print(f"  CV MAE: {mae_cv:.4f}")
            print(f"  CV RMSE: {rmse_cv:.4f}")

            results[name] = {
                'Best Estimator': best_estimator,
                'Best CV R2': best_score,
                'Best Params': best_params,
                'CV MAE': mae_cv,
                'CV RMSE': rmse_cv
            }

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_model_name = name
                best_overall_estimator = best_estimator

        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'Error': str(e)}

    print(f"\nBest overall model based on CV R2: {best_overall_model_name} (R2 = {best_overall_score:.4f})")

    # Save results summary
    results_summary = {name: {k: v for k, v in data.items() if k != 'Best Estimator'}
                       for name, data in results.items() if 'Error' not in data}
    results_df = pd.DataFrame.from_dict(results_summary, orient='index')
    results_df = results_df.sort_values(by='Best CV R2', ascending=False)
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path)
    print(f"Model comparison results saved to: {results_path}")

    return best_overall_model_name, best_overall_estimator, results

def plot_feature_importance(model, feature_names, output_dir, model_name):
    """Extracts and plots feature importance for tree-based models."""
    print("\n--- Calculating Feature Importance ---")
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not support feature_importances_ attribute. Skipping.")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Save importance data
    importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance data saved to: {importance_path}")

    # Plot importance
    plt.figure(figsize=(12, max(6, len(feature_names) * 0.3))) # Adjust height based on number of features
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {plot_path}")

    return importance_df

def perform_shap_analysis(model, X_rfe, feature_names, output_dir, model_name):
    """Performs SHAP analysis and generates plots."""
    print("\n--- Performing SHAP Analysis ---")
    start_time = time.time()

    try:
        # Choose appropriate explainer
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, xgb.XGBRegressor)):
            print("Using TreeExplainer...")
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (SVR, GaussianProcessRegressor, Lasso, Ridge)):
             # KernelExplainer can be slow, especially for large datasets
             # Consider using a subset of data for background distribution if needed
             print(f"Using KernelExplainer for {model_name} (this might take a while)...")
             # Use median or a sample for background data to speed up
             background_data = shap.sample(X_rfe, min(100, X_rfe.shape[0])) # Use a sample or median
             explainer = shap.KernelExplainer(model.predict, background_data)
        # Add LinearExplainer if needed, though Tree/Kernel cover most cases
        # elif isinstance(model, (Lasso, Ridge)):
        #     print("Using LinearExplainer...")
        #     explainer = shap.LinearExplainer(model, X_rfe)
        else:
            print(f"Warning: SHAP explainer not explicitly defined for model type {type(model)}. Attempting KernelExplainer.")
            background_data = shap.sample(X_rfe, min(100, X_rfe.shape[0]))
            explainer = shap.KernelExplainer(model.predict, background_data)

        # Calculate SHAP values (use X_rfe DataFrame for feature names)
        shap_values = explainer.shap_values(X_rfe)

        end_time = time.time()
        print(f"SHAP values calculated in {end_time - start_time:.2f} seconds.")

        # Ensure shap_values is compatible with plotting functions
        # KernelExplainer might return a single array, TreeExplainer might too for single output regression
        if isinstance(shap_values, list) and len(shap_values) == 1:
             shap_values = shap_values[0] # Adjust if explainer returns list for single output

        # Create SHAP Explanation object if needed (newer SHAP versions prefer this)
        try:
             shap_explanation = shap.Explanation(values=shap_values,
                                                  base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0, # Provide base value if available
                                                  data=X_rfe.values, # Use numpy array for data
                                                  feature_names=feature_names)
        except Exception:
             # Fallback for older versions or if Explanation fails
             shap_explanation = shap_values # Use raw values


        # 1. SHAP Summary Plot (Beeswarm)
        print("Generating SHAP summary plot...")
        plt.figure()
        try:
             shap.summary_plot(shap_explanation, X_rfe, plot_type="dot", show=False)
             plt.title(f'SHAP Summary Plot ({model_name})')
             plt.tight_layout()
             plot_path = os.path.join(output_dir, f'{model_name}_shap_summary.png')
             plt.savefig(plot_path)
             plt.close()
             print(f"SHAP summary plot saved to: {plot_path}")
        except Exception as e:
             print(f"Error generating SHAP summary plot: {e}")
             plt.close()


        # 2. SHAP Feature Importance Plot (based on mean absolute SHAP)
        print("Generating SHAP feature importance plot...")
        plt.figure()
        try:
             shap.summary_plot(shap_explanation, X_rfe, plot_type="bar", show=False)
             plt.title(f'SHAP Feature Importance ({model_name})')
             plt.tight_layout()
             plot_path = os.path.join(output_dir, f'{model_name}_shap_feature_importance.png')
             plt.savefig(plot_path)
             plt.close()
             print(f"SHAP feature importance plot saved to: {plot_path}")
        except Exception as e:
             print(f"Error generating SHAP feature importance plot: {e}")
             plt.close()

        # (Optional) Dependence Plots for top N features
        # n_dependence_plots = 5
        # print(f"Generating SHAP dependence plots for top {n_dependence_plots} features...")
        # importance_shap_df = pd.DataFrame({
        #     'Feature': feature_names,
        #     'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
        # }).sort_values(by='MeanAbsSHAP', ascending=False)
        #
        # for feature in importance_shap_df['Feature'].head(n_dependence_plots):
        #     try:
        #         plt.figure()
        #         shap.dependence_plot(feature, shap_values, X_rfe, interaction_index="auto", show=False)
        #         plt.title(f'SHAP Dependence Plot for {feature} ({model_name})')
        #         plt.tight_layout()
        #         plot_path = os.path.join(output_dir, f'{model_name}_shap_dependence_{feature}.png')
        #         plt.savefig(plot_path)
        #         plt.close()
        #         print(f"  Saved dependence plot for: {feature}")
        #     except Exception as e:
        #         print(f"Error generating SHAP dependence plot for {feature}: {e}")
        #         plt.close()

    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run Model-Based Feature Analysis")
    parser.add_argument('--feature_csv', required=False, default=r'results\filterd_features.csv', help='Path to input CSV file with filtered features (e.g., output of run_analysis.py).')
    parser.add_argument('--external_csv', required=False, default=r'data\database-250211.csv', help='Path to external data CSV (containing target variable).')
    parser.add_argument('--target_col', default=DEFAULT_TARGET_COL, help=f'Name of the target variable column (default: {DEFAULT_TARGET_COL}).')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--cv_folds', type=int, default=DEFAULT_CV_FOLDS, help=f'Number of folds for cross-validation (default: {DEFAULT_CV_FOLDS}).')
    parser.add_argument('--rfe_step', type=int, default=DEFAULT_RFE_STEP, help=f'Step size for RFECV (default: {DEFAULT_RFE_STEP}).')
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE, help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved in: {args.output_dir}")

    # Set random seed
    set_seed(args.random_state)
    print(f"Using random state: {args.random_state}")

    overall_start_time = time.time()

    # 1. Load and Prepare Data
    X, y, original_feature_names, preprocessor = load_and_prepare_data(
        args.feature_csv, args.external_csv, args.target_col
    )

    # 2. Perform RFECV
    selected_mask, selected_features, rfecv_selector = perform_rfecv(
        X, y, args.cv_folds, args.rfe_step, args.random_state
    )
    # Save RFECV plot
    rfecv_plot_path = os.path.join(args.output_dir, 'rfecv_performance.png')
    try:
         # Re-plot here or save the figure object from the function
         plt.figure(figsize=(10, 6))
         plt.xlabel("Number of features selected")
         plt.ylabel("Cross validation score (R^2)")
         if hasattr(rfecv_selector, 'grid_scores_'):
              scores = rfecv_selector.grid_scores_
         elif hasattr(rfecv_selector, 'cv_results_') and 'mean_test_score' in rfecv_selector.cv_results_:
              scores = rfecv_selector.cv_results_['mean_test_score']
         else: scores = None
         if scores is not None:
              plt.plot(range(1, len(scores) + 1), scores)
              plt.title('RFECV Performance vs. Number of Features')
              plt.grid(True)
              plt.scatter(rfecv_selector.n_features_, scores[rfecv_selector.n_features_-1], s=100,
                          facecolors='none', edgecolors='r', label=f'Optimal ({rfecv_selector.n_features_} features)')
              plt.legend()
              plt.savefig(rfecv_plot_path)
              plt.close()
              print(f"RFECV performance plot saved to: {rfecv_plot_path}")
         else:
              plt.close()
    except Exception as e:
         print(f"Warning: Could not save RFECV plot: {e}")
         plt.close()


    # Save selected features list
    selected_features_path = os.path.join(args.output_dir, 'rfecv_selected_features.txt')
    with open(selected_features_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected features list saved to: {selected_features_path}")

    # Filter data using selected features
    X_rfe = X[selected_features]

    # 3. Train and Evaluate Models
    best_model_name, best_estimator, all_model_results = train_and_evaluate_models(
        X_rfe, y, selected_features, args.cv_folds, args.random_state, args.output_dir
    )

    if best_estimator is None:
        print("\nError: No model could be trained successfully. Exiting.")
        sys.exit(1)

    # 4. Save the Best Model
    best_model_path = os.path.join(args.output_dir, f'best_model_{best_model_name}.joblib')
    joblib.dump(best_estimator, best_model_path)
    print(f"\nBest model ({best_model_name}) saved to: {best_model_path}")

    # 5. Feature Importance (for the best model)
    plot_feature_importance(best_estimator, selected_features, args.output_dir, best_model_name)

    # 6. SHAP Analysis (for the best model)
    perform_shap_analysis(best_estimator, X_rfe, selected_features, args.output_dir, best_model_name)

    overall_end_time = time.time()
    print(f"\n--- Model Analysis Pipeline Complete ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
