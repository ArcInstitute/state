import anndata as ad
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def load_folds_data(pattern, model_name=None):
    """
    Load all CSV files matching a glob pattern and combine them into a single DataFrame.
    Parameters:
    -----------
    pattern : str
        Glob pattern to match CSV files across different folds
    model_name : str, optional
        Name of the model to add as a column in the resulting DataFrame
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with data from all folds
    """
    # Get a list of all the CSV files matching the pattern
    fold_files = glob.glob(pattern)
    print(f"Found {len(fold_files)} files for {model_name}: {fold_files}")
    
    # Create an empty list to store DataFrames from each fold
    all_fold_results = []
    # Read each CSV file and append to the list
    for file in fold_files:
        if os.path.exists(file):
            fold_df = pd.read_csv(file)
            if model_name:
                fold_df["model"] = model_name
            all_fold_results.append(fold_df)
        else:
            print(f"Warning: File not found: {file}")
    
    if not all_fold_results:
        print(f"No data found for {model_name}")
        return pd.DataFrame()
    
    all_fold_results = pd.concat(all_fold_results, ignore_index=True)
    
    # attach additional metrics
    data_folder = pattern.split("/eval")[0]
    all_ct_dfs = []
    for file in glob.glob(os.path.join(data_folder, "*_downstream_de_results.csv")):
        if os.path.exists(file):
            ct_df = pd.read_csv(file)
            ct = file.split("_downstream_de_results.csv")[0].split("/")[-1]
            if "_new" in ct:
                ct = ct.replace("_new", "")
            # append as a row to the dataframe
            new_rows = pd.DataFrame(
                {
                    "celltype": [ct] * 4,
                    "metric_name": ["spearman", "pr_auc", "roc_auc", "significant_genes_count"],
                    "metric_val": [
                        ct_df["spearman"].mean(),
                        ct_df["pr_auc"].mean(),
                        ct_df["roc_auc"].mean(),
                        ct_df["significant_genes_count"].mean(),
                    ],
                    "model": [model_name] * 4,
                }
            )
            all_ct_dfs.append(new_rows)
    
    if all_ct_dfs:  # Check if the list is not empty before concatenating
        all_ct_df = pd.concat(all_ct_dfs, ignore_index=True)
        all_fold_results = pd.concat([all_fold_results, all_ct_df], ignore_index=True)
    
    return all_fold_results

# Define all 8 experiments and their readable names
experiments = {
    "hvg": "HVG",
    "pca": "PCA", 
    "scfound": "scFoundation",
    "scgpt": "scGPT",
    "tf": "TF",
    "uce": "UCE",
    "vci_1.4.2": "VCI_1.4.2",
    "vci_1.4.4": "VCI_1.4.4",
    "vci_1.5": "VCI_1.5",
    "vci_1.5.1": "VCI_1.5.1",
    "vci_1.5.2": "VCI_1.5.2",
}

# Base path
# base_path = "/large_storage/ctc/userspace/rohankshah/state_set_checkpoints/replogle"
base_path = "/large_storage/ctc/userspace/dhruvgautam/qwen/tahoe_vci_1.4.4_70m_mha/"

print("Loading data for all experiments...")
all_experiment_data = {}

# Load data for each experiment
for exp_name, display_name in experiments.items():
    if exp_name in ["srivatsan_all", "srivatsan_all_pca", "srivatsan_all_hvg"]:
        pattern = f"{base_path}/{exp_name}/mha_fold*/eval/map_random/step=56000/metrics.csv"
    else:
        pattern = f"{base_path}/{exp_name}/mha_fold*/eval/map_random/step=56000/metrics.csv"
    print(f"\nLoading {display_name} data...")
    exp_data = load_folds_data(pattern, display_name)
    
    if not exp_data.empty:
        all_experiment_data[display_name] = exp_data
    else:
        print(f"No data loaded for {display_name}")

# Calculate aggregated metrics for each experiment
experiment_metrics = []
for model_name, data in all_experiment_data.items():
    if not data.empty:
        metrics = data.groupby("metric_name")["metric_val"].mean().reset_index()
        metrics["model"] = model_name
        experiment_metrics.append(metrics)

if not experiment_metrics:
    print("No metrics data found for any experiment!")
    exit(1)

# Combine all metrics into a single dataframe
all_metrics = pd.concat(experiment_metrics, ignore_index=True)

# Create a pivot table for easier plotting
metrics_pivot = all_metrics.pivot(index="metric_name", columns="model", values="metric_val")

print("\nMetrics pivot table:")
print(metrics_pivot)

# Define specific metrics to include
target_metrics = [
    "DE_pval_fc_avg_N", 
    "DE_sig_genes_spearman", 
    "clustering_agreement", 
    "pearson_delta_cell_type", 
    "perturbation_score", 
    "pr_auc", 
    "roc_auc", 
    "spearman", 
    "DE_patk_pval_fc_av_200"
]

# Filter to only include target metrics that exist in the data
shared_metrics = [metric for metric in target_metrics if metric in metrics_pivot.index]

print(f"\nTarget metrics found in data: {shared_metrics}")
print(f"Missing metrics: {set(target_metrics) - set(shared_metrics)}")

# Define the model names in order
model_names = [name for name in experiments.values() if name in all_experiment_data.keys()]

print(f"Models with data: {model_names}")

# Plot settings
num_metrics = len(shared_metrics)
num_cols = 4
num_rows = (num_metrics + num_cols - 1) // num_cols  # Ceiling division

plt.figure(figsize=(20, num_rows * 5))

# Create subplots for each metric
for i, metric in enumerate(shared_metrics):
    plt.subplot(num_rows, num_cols, i + 1)
    
    # Extract values for this metric across all models (handle missing values)
    values = []
    model_names_for_metric = []
    
    for model in model_names:
        if model in metrics_pivot.columns and metric in metrics_pivot.index:
            value = metrics_pivot.loc[metric, model]
            if pd.notna(value):  # Only include non-NaN values
                values.append(value)
                model_names_for_metric.append(model)
    
    if values:  # Only plot if we have data
        # Create bar plot
        bars = plt.bar(range(len(values)), values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
        
        # Add value labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", 
                    ha="center", va="bottom", rotation=0, fontsize=8)
        
        plt.title(metric, fontsize=12)
        plt.xticks(range(len(model_names_for_metric)), model_names_for_metric, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
    else:
        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(metric, fontsize=12)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.suptitle("Comparison of Metrics Across All Srivatsan Experiments (step=best)", fontsize=16)

# Save the figure
plt.savefig("srivatsan_all_experiments_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Print summary statistics
print(f"\nSummary:")
print(f"Total experiments loaded: {len(all_experiment_data)}")
print(f"Total shared metrics: {len(shared_metrics)}")
print(f"Experiments: {list(all_experiment_data.keys())}")

# Create a summary table
summary_data = []
for model in model_names:
    if model in all_experiment_data:
        model_data = all_experiment_data[model]
        summary_data.append({
            'Model': model,
            'Total_Metrics': len(model_data['metric_name'].unique()),
            'Sample_Size': len(model_data)
        })

summary_df = pd.DataFrame(summary_data)
print(f"\nSummary by model:")
print(summary_df)