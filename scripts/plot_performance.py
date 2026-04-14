import argparse
import matplotlib.pyplot as plt
import os
import re

# Use LaTeX for text rendering
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# Layer type display names
LAYER_NAMES = {
    "Conv2D": r"Conv2D",
    "BatchNorm2D": r"BN",
    "ReLU": r"ReLU",
    "MaxPool2D": r"MaxPool",
    "Flatten": r"Flatten",
    "Dense": r"Dense",
    "Dropout": r"Dropout",
    "Softmax": r"Softmax",
}

# All 4 configurations (order: worst to best)
ALL_CONFIGS = ["OIANet_1_1_1_1", "OIANet_1_0_1_1", "OIANet_1_0_0_1", "OIANet_1_0_0_0"]

# Custom labels for x-axis (edita estos valores para personalizar los nombres en las gráficas)
CUSTOM_X_LABELS = {
    "1_0_0_0": "Cython maxpool2D",  # Cambiar a lo que quieras, ej: "Baseline"
    "1_0_0_1": "Dense GEMM",  # Cambiar a lo que quieras, ej: "Con MaxPool"
    "1_0_1_1": "Img2colfused",  # Cambiar a lo que quieras, ej: "Con Dense"
    "1_1_1_1": "Baseline",  # Cambiar a lo que quieras, ej: "Todas"
}

# Font sizes for different elements
FONTSIZE_TITLE = 14
FONTSIZE_AXIS_LABEL = 18
FONTSIZE_X_LABELS = 18
FONTSIZE_Y_LABELS = 10
FONTSIZE_PERCENTAGE = 14
FONTSIZE_LEGEND = 12

# Presets for different configurations
PRESETS = {
    1: {"configs": [ALL_CONFIGS[0]], "name": "1"},
    2: {"configs": [ALL_CONFIGS[0], ALL_CONFIGS[1]], "name": "2"},
    3: {"configs": [ALL_CONFIGS[0], ALL_CONFIGS[1], ALL_CONFIGS[2]], "name": "3"},
    4: {"configs": ALL_CONFIGS, "name": "4"},
}


def parse_output_file(filepath):
    """Parse the output file and extract FW and BW layer performance data."""
    fw_layers = []
    bw_layers = []
    current_section = None
    ips_value = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("FW Layer;"):
                current_section = "fw"
                continue
            elif line.startswith("BW Layer;"):
                current_section = "bw"
                continue
            elif line.startswith("===="):
                current_section = None
                continue
            elif line.startswith("Total time:"):
                # Extract IPS value from "Total time: X.XXs IPS: X.XXimages/sec"
                match = re.search(r"IPS:\s*([\d.]+)", line)
                if match:
                    ips_value = float(match.group(1))
                continue

            if current_section and ";" in line:
                parts = line.split(";")
                if len(parts) == 4:
                    name, batch, time_s, perf = parts
                    fw_layers.append(
                        (name, float(time_s))
                    ) if current_section == "fw" else bw_layers.append(
                        (name, float(time_s))
                    )

    return fw_layers, bw_layers, ips_value


def make_unique_names(layers):
    """Add index to layer names to distinguish repeated layers (e.g. Conv2D_1, Conv2D_2)."""
    counts = {}
    unique_layers = []
    for name, t in layers:
        counts[name] = counts.get(name, 0) + 1
        unique_layers.append((f"{name}_{counts[name]}", t))
    return unique_layers


def compute_percentages(layers):
    """Compute percentage of total time for each layer."""
    total_time = sum(t for _, t in layers)
    if total_time == 0:
        return [], 0.0
    pcts = [(name, (t / total_time) * 100) for name, t in layers]
    return pcts, total_time


def plot_stacked_bar(fw_layers, bw_layers, filename, output_dir):
    """Generate a vertical stacked bar showing cumulative % time per individual layer."""
    fw_layers = make_unique_names(fw_layers)
    bw_layers = make_unique_names(bw_layers)

    fw_pcts, fw_total = compute_percentages(fw_layers) if fw_layers else ([], 0.0)
    bw_pcts, bw_total = compute_percentages(bw_layers) if bw_layers else ([], 0.0)

    # Collect all unique layer names preserving order
    all_names = [n for n, _ in fw_layers] + [n for n, _ in bw_layers]
    all_names = list(dict.fromkeys(all_names))

    # Color map: same base type gets the same color
    base_types = list(dict.fromkeys(n.rsplit("_", 1)[0] for n in all_names))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(base_types))
    base_colors = {bt: cmap(i) for i, bt in enumerate(base_types)}
    colors = {n: base_colors[n.rsplit("_", 1)[0]] for n in all_names}

    bar_labels = []
    bar_data = []
    if fw_pcts:
        bar_labels.append(f"{filename} FW")
        bar_data.append(dict(fw_pcts))
    if bw_pcts:
        bar_labels.append(f"{filename} BW")
        bar_data.append(dict(bw_pcts))

    fig, ax = plt.subplots(figsize=(max(4, len(bar_labels) * 3), 7))

    for bar_idx, (label, pct_dict) in enumerate(zip(bar_labels, bar_data)):
        bottom = 0.0
        for layer_name in all_names:
            pct = pct_dict.get(layer_name, 0.0)
            if pct > 0:
                ax.bar(
                    bar_idx,
                    pct,
                    bottom=bottom,
                    color=colors[layer_name],
                    edgecolor="none",
                    linewidth=0,
                )
                if pct > 5:
                    ax.text(
                        bar_idx,
                        bottom + pct / 2,
                        f"{layer_name}\n{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                bottom += pct

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=FONTSIZE_X_LABELS)
    ax.set_ylabel(r"Time (\%)", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title(
        f"Layer Time Distribution $-$ {filename}", fontsize=FONTSIZE_TITLE, pad=50
    )
    ax.tick_params(axis="y", labelsize=FONTSIZE_Y_LABELS)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Legend: one entry per base type, placed just below the title
    base_type_labels = [LAYER_NAMES.get(bt, bt) for bt in base_types]
    handles = [plt.Rectangle((0, 0), 1, 1, color=base_colors[bt]) for bt in base_types]
    ax.legend(
        handles,
        base_type_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(6, len(base_types)),
        fontsize=FONTSIZE_LEGEND,
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


def plot_stacked_layers_comparison(file_list, outs_dir, output_dir, preset_name=""):
    """Generate a stacked bar chart comparing layer percentages across configurations."""
    all_layers_data = {}  # {layer_name: {config_name: percentage}}
    config_names = []

    for filename in file_list:
        filepath = os.path.join(outs_dir, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        config_name = filename.replace("OIANet_", "")
        config_names.append(config_name)

        fw_layers, bw_layers, _ = parse_output_file(filepath)

        # Combine FW and BW layers
        all_layers = fw_layers + bw_layers
        all_layers = make_unique_names(all_layers)
        pcts, total_time = compute_percentages(all_layers)

        # Store percentages for each layer
        for layer_name, pct in pcts:
            if layer_name not in all_layers_data:
                all_layers_data[layer_name] = {}
            all_layers_data[layer_name][config_name] = pct

    if not config_names:
        print(f"No files found for preset")
        return

    # Get all unique layer names in order
    all_layer_names = list(all_layers_data.keys())

    # Get unique base types for colors
    base_types = list(dict.fromkeys(n.rsplit("_", 1)[0] for n in all_layer_names))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(base_types))
    base_colors = {bt: cmap(i) for i, bt in enumerate(base_types)}
    colors = {n: base_colors[n.rsplit("_", 1)[0]] for n in all_layer_names}

    # Create figure with dynamic sizing
    n_bars = len(config_names)
    if n_bars == 1:
        fig_width = 5
    elif n_bars == 2:
        fig_width = 7
    else:
        fig_width = max(8, n_bars * 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Create stacked bar chart
    x_pos = range(len(config_names))
    bottom = [0.0] * len(config_names)

    for layer_name in all_layer_names:
        layer_pcts = [
            all_layers_data[layer_name].get(config, 0.0) for config in config_names
        ]

        ax.bar(
            x_pos,
            layer_pcts,
            bottom=bottom,
            label=layer_name,
            color=colors[layer_name],
            edgecolor="none",
            linewidth=0,
        )

        # Add percentage labels
        for i, (pct, b) in enumerate(zip(layer_pcts, bottom)):
            if pct > 3:  # Only show labels for segments > 3%
                ax.text(
                    i,
                    b + pct / 2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=FONTSIZE_PERCENTAGE,
                )

        # Update bottom for next layer
        bottom = [b + p for b, p in zip(bottom, layer_pcts)]

    ax.set_xticks(x_pos)
    # Use custom labels if available, otherwise fall back to config names
    custom_labels = [CUSTOM_X_LABELS.get(name, name) for name in config_names]
    ax.set_xticklabels(
        custom_labels, rotation=45, ha="right", fontsize=FONTSIZE_X_LABELS
    )
    ax.set_ylabel(r"Time (\%)", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title(
        f"Layer Percentage Distribution $-$ {preset_name}",
        fontsize=FONTSIZE_TITLE,
        pad=15,
    )
    ax.tick_params(axis="y", labelsize=FONTSIZE_Y_LABELS)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Legend: one entry per base type with display names
    base_type_labels = [LAYER_NAMES.get(bt, bt) for bt in base_types]
    handles = [plt.Rectangle((0, 0), 1, 1, color=base_colors[bt]) for bt in base_types]
    ax.legend(
        handles,
        base_type_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        ncol=1,
        fontsize=FONTSIZE_LEGEND,
    )

    plt.subplots_adjust(right=0.85)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"preset{preset_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Layers comparison plot saved to {out_path}")
    plt.close()


def plot_performance_metrics(file_list, outs_dir, output_dir):
    """Generate a line plot with execution time and imgs/s for all configurations."""
    times = []
    ips_values = []
    config_names = []

    for filename in file_list:
        filepath = os.path.join(outs_dir, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        config_name = filename.replace("OIANet_", "")
        config_names.append(config_name)

        fw_layers, bw_layers, ips_value = parse_output_file(filepath)

        # Calculate total time
        all_layers = fw_layers + bw_layers
        total_time = sum(t for _, t in all_layers)
        times.append(total_time)
        ips_values.append(ips_value if ips_value else 0)

    if not config_names:
        print("No files found for performance metrics")
        return

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Prepare x-axis
    x_pos = range(len(config_names))
    custom_labels = [CUSTOM_X_LABELS.get(name, name) for name in config_names]

    # Plot time on left y-axis
    color1 = "#1f77b4"  # Blue
    ax1.set_xlabel(r"Configuration", fontsize=FONTSIZE_AXIS_LABEL)
    ax1.set_ylabel(r"Execution Time (s)", fontsize=FONTSIZE_AXIS_LABEL)
    line1 = ax1.plot(
        x_pos,
        times,
        color=color1,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Execution Time",
    )
    ax1.tick_params(axis="y", labelsize=FONTSIZE_Y_LABELS)
    ax1.tick_params(axis="x", labelsize=FONTSIZE_X_LABELS)

    # Create right y-axis for IPS
    ax2 = ax1.twinx()
    color2 = "#ff7f0e"  # Orange
    ax2.set_ylabel(r"Images/sec", fontsize=FONTSIZE_AXIS_LABEL)
    line2 = ax2.plot(
        x_pos,
        ips_values,
        color=color2,
        marker="s",
        linewidth=2.5,
        markersize=8,
        label="IPS",
    )
    ax2.tick_params(axis="y", labelsize=FONTSIZE_Y_LABELS)

    # Set x-axis labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        custom_labels, rotation=45, ha="right", fontsize=FONTSIZE_X_LABELS
    )

    # Title
    ax1.set_title(
        r"Performance Metrics Across Configurations", fontsize=FONTSIZE_TITLE, pad=20
    )

    # Add grid
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # Legend: combine both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=FONTSIZE_LEGEND)

    plt.subplots_adjust(right=0.9)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "performance_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Performance metrics plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot layer performance comparison.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single output file (e.g. outs/OIANet_1_0_0_0)",
    )
    parser.add_argument(
        "--layers-all",
        action="store_true",
        help="Plot layer percentage distribution for all preset configurations",
    )
    parser.add_argument(
        "--outs_dir",
        type=str,
        default="outs",
        help="Directory containing output files (default: outs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save the plots (default: plots)",
    )
    args = parser.parse_args()

    if args.layers_all:
        print("Generating layer comparisons for all presets...")
        for preset_id, preset_info in PRESETS.items():
            print(f"  Preset {preset_id}: {preset_info['configs']}")
            plot_stacked_layers_comparison(
                preset_info["configs"],
                args.outs_dir,
                args.output_dir,
                preset_name=preset_info["name"],
            )
        # Also generate performance metrics plot
        print("Generating performance metrics plot...")
        plot_performance_metrics(
            ALL_CONFIGS,
            args.outs_dir,
            args.output_dir,
        )
    elif args.file:
        # Plot single file
        filename = os.path.basename(args.file)
        fw, bw, ips = parse_output_file(args.file)
        if not fw:
            print(f"No forward layer data found in {args.file}")
        else:
            plot_stacked_bar(fw, bw, filename, args.output_dir)
    else:
        print("Please provide either --layers-all or --file option")
        print("\nExamples:")
        print("  python plot_performance.py --layers-all")
        print("  python plot_performance.py --file outs/OIANet_1_0_0_0")
