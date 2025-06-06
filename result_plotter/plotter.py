from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import re

# 1) plot cuda base speedup
# 2) plot cuda halo vs base speedup
# 3) plot cuda extern vs halo speedup
# 4) plot all speedups in one plot including openmp  

# open the times file and read the data
def read_csv(file_path: os.path) -> pd.Series:
    try:
        data = pd.read_csv(file_path, header=None, names=["time"])
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return pd.DataFrame()
    
def plot_speedup(data: pd.DataFrame, title: str, line_label: str, xlabel: str, ylabel: str, save_path: str, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    plt.plot(data["ds"], data["speedup"], marker='o', label=line_label, color='blue')

    additional_lines = kwargs.get('additional_lines')
    if additional_lines:
        for line_info in additional_lines:
            df = line_info.get('data')
            label = line_info.get('label')
            color = line_info.get('color') # Optional: allow specifying color
            marker = line_info.get('marker', 'x') # Optional: allow specifying marker
            if df is not None and label is not None and "ds" in df.columns and "speedup" in df.columns:
                plt.plot(df["ds"], df["speedup"], marker=marker, label=label, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(data["ds"])  # Ensure all ds values are shown on x-axis
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # get the path to this file 
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # for each folder in the script path, read the csv files and plot the data
    for folder in os.listdir(script_dir):
        if not os.path.isdir(os.path.join(script_dir, folder)):
            continue

        plot_dir = os.path.join(script_dir, folder, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        folder_summary_df = pd.DataFrame()

        folder_path = os.path.join(script_dir, folder)

        # read the csv files in the folder
        for ds_folder in os.listdir(folder_path):
            if ds_folder == "plots": 
                continue

            for file_name in os.listdir(os.path.join(folder_path, ds_folder)):
                if not file_name.endswith(".txt") and not file_name.endswith(".csv"):
                    continue
                current_file_path = os.path.join(folder_path, ds_folder, file_name)
                
                data = read_csv(current_file_path)
                file_name = file_name[:-4] 

                if data.empty:
                    continue

                # add the mean to a dataframe
                mean = {
                    "file": file_name,
                    "ds": int(re.search(r'\d+', ds_folder).group(0)),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                }

                new_row_df = pd.DataFrame([mean])
                folder_summary_df = pd.concat([folder_summary_df, new_row_df], ignore_index=True)

        folder_summary_df.to_csv(os.path.join(plot_dir, f"{folder}_summary.csv"), index=False)

        # 1) plot cuda base speedup
        cuda_base_df  = folder_summary_df[folder_summary_df["file"] == "cuda_base_results"].copy()
        sequential = folder_summary_df[folder_summary_df["file"] == "seq_results"]
        
        cuda_base_df.set_index("ds", inplace=True)
        sequential.set_index("ds", inplace=True)

        cuda_base_df["speedup"] = sequential["mean"].values / cuda_base_df["mean"].values
        cuda_base_df.reset_index(inplace=True)
        cuda_base_df.sort_values(by="ds", inplace=True)

        plot_speedup(
            data=cuda_base_df,
            title=f"Speedup for {folder} kernel",
            xlabel="Downsample Factor",
            ylabel="Speedup",
            line_label="Cuda Base Speedup",
            save_path=os.path.join(plot_dir, f"{folder}_speedup_base.png")
        )

        # 2) plot cuda halo vs base speedup
        cuda_halo_df = folder_summary_df[folder_summary_df["file"] == "cuda_halo_results"].copy()
        cuda_halo_df.set_index("ds", inplace=True)

        cuda_halo_df["speedup"] = sequential["mean"].values / cuda_halo_df["mean"].values
        cuda_halo_df.reset_index(inplace=True)
        cuda_halo_df.sort_values(by="ds", inplace=True)

        plot_speedup(
            data=cuda_halo_df,
            title=f"Speedup for {folder} kernel with Halo Effect",
            xlabel="Downsample Factor",
            ylabel="Speedup",
            line_label="Cuda Halo Speedup",
            save_path=os.path.join(plot_dir, f"{folder}_speedup_halo.png"),
            additional_lines=[{
                'data': cuda_base_df,
                'label': 'Base Speedup',
                'color': 'grey',
                'marker': '.'
            }]
        )

        # 3) plot cuda extern vs halo speedup
        cuda_extern_df = folder_summary_df[folder_summary_df["file"] == "cuda_extern_results"].copy()
        cuda_extern_df.set_index("ds", inplace=True)

        cuda_extern_df["speedup"] = sequential["mean"].values / cuda_extern_df["mean"].values
        cuda_extern_df.reset_index(inplace=True)
        cuda_extern_df.sort_values(by="ds", inplace=True)

        plot_speedup(
            data=cuda_extern_df,
            title=f"Speedup for {folder} kernel with Shared Memory",
            xlabel="Downsample Factor",
            ylabel="Speedup",
            line_label="Cuda Shared Memory Speedup",
            save_path=os.path.join(plot_dir, f"{folder}_speedup_extern.png"),
            additional_lines=[{
                'data': cuda_halo_df,
                'label': 'Halo Speedup',
                'color': 'grey',
                'marker': '.'
            }]
        )

        # 4) plot all speedups in one plot including openmp
        openmp_df = folder_summary_df[folder_summary_df["file"] == "parallel_results"].copy()
        openmp_df.set_index("ds", inplace=True)

        openmp_df["speedup"] = sequential["mean"].values / openmp_df["mean"].values
        openmp_df.reset_index(inplace=True)
        openmp_df.sort_values(by="ds", inplace=True)

        plot_speedup(
            data=openmp_df,
            title=f"Speedup for {folder} kernel with OpenMP",
            xlabel="Downsample Factor",
            ylabel="Speedup",
            line_label="OpenMP Speedup",
            save_path=os.path.join(plot_dir, f"{folder}_speedup_openmp.png"),
            additional_lines=[
                {
                    'data': cuda_base_df,
                    'label': 'Base Speedup',
                    'color': 'green',
                    'marker': 'o'
                },
                {
                    'data': cuda_halo_df,
                    'label': 'Halo Speedup',
                    'color': 'grey',
                    'marker': 'o'
                },
                {
                    'data': cuda_extern_df,
                    'label': 'Shared Memory Speedup',
                    'color': 'orange',
                    'marker': 'o'
                }
            ]
        )