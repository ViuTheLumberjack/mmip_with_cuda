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
    
def plot_times(data: pd.DataFrame, title: str, line_label: str, xlabel: str, ylabel: str, save_path: str, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    marker = kwargs.get('marker', 'o')  # Optional: allow specifying marker
    color = kwargs.get('color', 'blue')  # Optional: allow specifying color
    plt.plot(data["ds"], data["mean"], marker=marker, color=color, label=line_label)

    additional_lines = kwargs.get('additional_lines')
    if additional_lines:
        for line_info in additional_lines:
            df = line_info.get('data')
            label = line_info.get('label')
            color = line_info.get('color') # Optional: allow specifying color
            marker = line_info.get('marker', 'x') # Optional: allow specifying marker
            if df is not None and label is not None and "ds" in df.columns and "mean" in df.columns:
                plt.plot(df["ds"], df["mean"], marker=marker, label=label, color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(data.index)  # Ensure all indices are shown on x-axis
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
def plot_speedup(data: pd.DataFrame, title: str, line_label: str, xlabel: str, ylabel: str, save_path: str, *args, **kwargs):
    plt.figure(figsize=(10, 6))

    marker = kwargs.get('marker', 'o')  # Optional: allow specifying marker
    color = kwargs.get('color', 'blue')  # Optional: allow specifying color
    plt.plot(data["ds"], data["speedup"], marker=marker, label=line_label, color=color)

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

def plot_tiles(data: pd.DataFrame, title: str, ylabel: str, save_path: str):
    cuda_elements = [cuda for cuda in data["file"].unique() if "cuda" in cuda]
    #kernels = data["kernel"].unique()
    # plt.bar(data["tile_size"], data["speedup"], marker=marker, label=line_label, color=color)
    
    custom_kernel_order = ['cross3', 'cross7', 'cross15', 'cross33', 'rect3', 'rect7', 'rect15', 'rect33', 'ellipse3', 'ellipse7', 'ellipse15', 'ellipse33']
    def get_kernel_order(day):
        try:
            return custom_kernel_order.index(day)
        except ValueError:
            return len(custom_kernel_order)

    fig, ax = plt.subplots(5, 2, layout='constrained')
    fig.figure.set_size_inches(30, 40)
    x = np.arange(len(custom_kernel_order))  # the label locations
    width = 0.2  # the width of the bars

    for ds in data["ds"].unique():
        multiplier = 0
        if ds == 0:
            continue
        # filter the data for the current ds        
        for file in cuda_elements:
            offset = width * multiplier
            sort_df = data[(data["tile_size"] == 32) & (data["ds"] == ds) & (data["file"] == file)].sort_values(by='kernel', key=lambda x: x.apply(get_kernel_order))
            rects = ax[(ds-1)//2, 0 if ds%2==1 else 1].bar(x + offset, sort_df["speedup"], width, label=file)
            ax[(ds-1)//2, 0 if ds%2==1 else 1].bar_label(rects, padding=3, fmt='%d')
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[(ds-1)//2, 0 if ds%2==1 else 1].set_ylabel(ylabel)
        ax[(ds-1)//2, 0 if ds%2==1 else 1].set_title(title.format(f"(ds={ds})"))
        ax[(ds-1)//2, 0 if ds%2==1 else 1].set_xticks(x + width, custom_kernel_order)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.savefig(save_path.format(0))
    plt.close()

def plot_speedup_for_kernels(data: pd.DataFrame, title: str, ylabel: str, save_path: str):
    cuda_elements = [cuda for cuda in data["file"].unique() if "cuda" in cuda]
    custom_kernel_order = ['cross15', 'cross33', 'rect15', 'rect33', 'ellipse15', 'ellipse33']
    sizes = [15, 33]
        
    data_filtered = data[(data["kernel"].isin(custom_kernel_order)) & (data["ds"] == 1) & (data["tile_size"] <= 512)]
    v = plt.get_cmap('viridis')
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    fig.figure.set_size_inches(10, 10)

    for size in sizes:
        size_data = data_filtered[data_filtered["kernel"].str.contains(str(size))]
        ax[size//15 - 1].set_prop_cycle(color=[v(i) for i in np.linspace(0, 1, len(cuda_elements) * len(size_data["kernel"].unique()))])
        for file in cuda_elements:
            sort_df = size_data[size_data["file"] == file].sort_values(by='tile_size')
            for kernel in sort_df["kernel"].unique():
                sort_df_kernel = sort_df[sort_df["kernel"] == kernel]
                if sort_df_kernel.empty:
                    continue
                ax[size//15 - 1].plot(sort_df_kernel["tile_size"], sort_df_kernel["speedup"], marker='o', label=f"{kernel} ({file})")
        
        ax[size//15 - 1].set_xscale('log')
        ax[size//15 - 1].set_xticks(sort_df["tile_size"].unique())
        ax[size//15 - 1].set_xticklabels(sort_df["tile_size"].unique(), rotation=45)
        ax[size//15 - 1].set_xlabel("Tile Size")
        ax[size//15 - 1].set_ylabel(ylabel)
        ax[size//15 - 1].set_title(title.format(f"(kernel size={size})"))
        # legend outside the plo
        ax[size//15 - 1].legend(title="Cuda Elements", loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path.format(0))
    plt.close()

if __name__ == "__main__":
    # get the path to this file 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_data = pd.read_csv(os.path.join(script_dir, "all_data_summary.csv"))

    if all_data.empty:    # for each folder in the script path, read the csv files and plot the data
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
                    match = re.search(r'\d+', file_name)
                    tile_size = int(match.group(0)) if match else 0
                    file_name_cleaned = re.sub(r'_\d+', '', file_name) if match else file_name
                    
                    mean = {
                        "file": file_name_cleaned,
                        "tile_size": tile_size,
                        "kernel": folder,
                        "ds": int(re.search(r'\d+', ds_folder).group(0)),
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                        "min": float(data.min()),
                        "max": float(data.max()),
                    }

                    new_row_df = pd.DataFrame([mean])
                    folder_summary_df = pd.concat([folder_summary_df, new_row_df], ignore_index=True)

            folder_summary_df.to_csv(os.path.join(plot_dir, f"{folder}_summary.csv"), index=False)
            all_data = pd.concat([all_data, folder_summary_df], ignore_index=True)

            for tile_size in folder_summary_df["tile_size"].unique():
                if tile_size == 0:
                    continue
                # 1) plot cuda base speedup
                cuda_base_df  = folder_summary_df[(folder_summary_df["file"] == "cuda_base_results") & (folder_summary_df["tile_size"] == tile_size)].copy()
                sequential = folder_summary_df[(folder_summary_df["file"] == "seq_results") & (folder_summary_df["tile_size"] == 0)].copy()
                
                cuda_base_df.set_index("ds", inplace=True)
                sequential.set_index("ds", inplace=True)

                cuda_base_df["speedup"] = sequential["mean"].values / cuda_base_df["mean"].values
                cuda_base_df.reset_index(inplace=True)
                cuda_base_df.sort_values(by="ds", inplace=True)
                sequential.reset_index(inplace=True)
                sequential.sort_values(by="ds", inplace=True)

                plot_speedup(
                    data=cuda_base_df,
                    title=f"Speedup for {folder} kernel (Tile Size: {tile_size})",
                    xlabel="Downsample Factor",
                    ylabel="Speedup",
                    line_label="Cuda Base Speedup",
                    save_path=os.path.join(plot_dir, f"{folder}_{tile_size}_speedup_base.png"),
                    marker='o',
                    color='green'
                )

                # 2) plot cuda halo vs base speedup
                cuda_halo_df = folder_summary_df[(folder_summary_df["file"] == "cuda_halo_results") & (folder_summary_df["tile_size"] == tile_size)].copy()
                cuda_halo_df.set_index("ds", inplace=True)

                cuda_halo_df["speedup"] = sequential["mean"].values / cuda_halo_df["mean"].values
                cuda_halo_df.reset_index(inplace=True)
                cuda_halo_df.sort_values(by="ds", inplace=True)

                plot_speedup(
                    data=cuda_halo_df,
                    title=f"Speedup for {folder} kernel with Halo Effect (Tile Size: {tile_size})",
                    xlabel="Downsample Factor",
                    ylabel="Speedup",
                    line_label="Cuda Halo Speedup",
                    save_path=os.path.join(plot_dir, f"{folder}_{tile_size}_speedup_halo.png"),
                    marker='x',
                    color='grey',
                    additional_lines=[{
                        'data': cuda_base_df,
                        'label': 'Base Speedup',
                        'color': 'green',
                        'marker': '.'
                    }]
                )

                # 3) plot cuda extern vs halo speedup
                cuda_extern_df = folder_summary_df[(folder_summary_df["file"] == "cuda_extern_results") & (folder_summary_df["tile_size"] == tile_size)].copy()
                cuda_extern_df.set_index("ds", inplace=True)

                cuda_extern_df["speedup"] = sequential["mean"].values / cuda_extern_df["mean"].values
                cuda_extern_df.reset_index(inplace=True)
                cuda_extern_df.sort_values(by="ds", inplace=True)

                plot_speedup(
                    data=cuda_extern_df,
                    title=f"Speedup for {folder} kernel with Shared Memory (Tile Size: {tile_size})",
                    xlabel="Downsample Factor",
                    ylabel="Speedup",
                    line_label="Cuda Shared Memory Speedup",
                    save_path=os.path.join(plot_dir, f"{folder}_{tile_size}_speedup_extern.png"),
                    marker='x',
                    color='orange',
                    additional_lines=[{
                        'data': cuda_halo_df,
                        'label': 'Halo Speedup',
                        'color': 'grey',
                        'marker': 'x'
                    }]
                )

                # 4) plot all speedups in one plot including openmp
                openmp_df = folder_summary_df[(folder_summary_df["file"] == "parallel_results") & (folder_summary_df["tile_size"] == 0)].copy()
                openmp_df.set_index("ds", inplace=True)

                openmp_df["speedup"] = sequential["mean"].values / openmp_df["mean"].values
                openmp_df.reset_index(inplace=True)
                openmp_df.sort_values(by="ds", inplace=True)

                plot_speedup(
                    data=openmp_df,
                    title=f"Speedup for {folder} kernel with OpenMP (Tile Size: {tile_size})",
                    xlabel="Downsample Factor",
                    ylabel="Speedup",
                    line_label="OpenMP Speedup",
                    save_path=os.path.join(plot_dir, f"{folder}_{tile_size}_speedup_openmp.png"),
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

                plot_times(
                    data=cuda_base_df,
                    title=f"Time for {folder} kernel (Tile Size: {tile_size})",
                    xlabel="Downsample Factor",
                    ylabel="Time (microseconds)",
                    line_label="Base",
                    save_path=os.path.join(plot_dir, f"{folder}_{tile_size}_times.png"),
                    color='green',
                    marker='o',
                    additional_lines=[
                        {
                            'data': cuda_halo_df,
                            'label': 'Halo',
                            'color': 'grey',
                            'marker': 'o'
                        },
                        {
                            'data': cuda_extern_df,
                            'label': 'Shared Memory',
                            'color': 'orange',
                            'marker': 'o'
                        }
                    ]
                )
        all_data.to_csv(os.path.join(script_dir, "all_data_summary.csv"), index=False)
    
    kernels = all_data["kernel"].unique()
    # divide the data with tile_size greater by 0 by the entry with file=seq_results with the same kernel and ds
    for kernel in kernels:
        kernel_data = all_data[all_data["kernel"] == kernel]
        seq_data = kernel_data[kernel_data["file"] == "seq_results"]
        if seq_data.empty:
            continue
        # calculate the speedup for each tile size
        kernel_data = kernel_data[kernel_data["tile_size"] > 0]
        kernel_data["speedup"] = kernel_data.groupby(["tile_size", "ds"])["mean"].transform(
            lambda x: seq_data["mean"].values[0] / x
        )
        all_data.loc[kernel_data.index, "speedup"] = kernel_data["speedup"]
    
    # now with all the data
    # plot the speedup for different tile sizes on the original image
    plot_tiles(
        data=all_data,
        title="Tile Size Comparison for different kernel {}",
        ylabel="Speedup",
        save_path=os.path.join(script_dir, "tile_size_comparison_{}.png")
    )

    plot_speedup_for_kernels(
        data=all_data,
        title="Speedup for different kernel sizes {}",
        ylabel="Speedup",
        save_path=os.path.join(script_dir, "speedup_for_kernels_{}.png")
    )
