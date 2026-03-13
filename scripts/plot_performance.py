import argparse
import matplotlib.pyplot as plt
import os


def parse_output_file(filepath):
    """Parse the output file and extract FW and BW layer performance data."""
    fw_layers = []
    bw_layers = []
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("FW Layer;"):
                current_section = 'fw'
                continue
            elif line.startswith("BW Layer;"):
                current_section = 'bw'
                continue
            elif line.startswith("===="):
                current_section = None
                continue

            if current_section and ';' in line:
                parts = line.split(';')
                if len(parts) == 4:
                    name, batch, time_s, perf = parts
                    fw_layers.append((name, float(time_s))) if current_section == 'fw' else \
                        bw_layers.append((name, float(time_s)))

    return fw_layers, bw_layers


def make_unique_names(layers):
    """Add index to layer names to distinguish repeated layers (e.g. Conv2D_1, Conv2D_2)."""
    counts = {}
    unique_layers = []
    for name, t in layers:
        counts[name] = counts.get(name, 0) + 1
        unique_layers.append((f'{name}_{counts[name]}', t))
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
    base_types = list(dict.fromkeys(n.rsplit('_', 1)[0] for n in all_names))
    cmap = plt.colormaps.get_cmap('tab20').resampled(len(base_types))
    base_colors = {bt: cmap(i) for i, bt in enumerate(base_types)}
    colors = {n: base_colors[n.rsplit('_', 1)[0]] for n in all_names}

    bar_labels = []
    bar_data = []
    if fw_pcts:
        bar_labels.append(f'{filename} FW')
        bar_data.append(dict(fw_pcts))
    if bw_pcts:
        bar_labels.append(f'{filename} BW')
        bar_data.append(dict(bw_pcts))

    fig, ax = plt.subplots(figsize=(max(4, len(bar_labels) * 3), 7))

    for bar_idx, (label, pct_dict) in enumerate(zip(bar_labels, bar_data)):
        bottom = 0.0
        for layer_name in all_names:
            pct = pct_dict.get(layer_name, 0.0)
            if pct > 0:
                ax.bar(bar_idx, pct, bottom=bottom, color=colors[layer_name],
                       edgecolor='none', linewidth=0)
                if pct > 5:
                    ax.text(bar_idx, bottom + pct / 2, f'{layer_name}\n{pct:.1f}%',
                            ha='center', va='center', fontsize=7, fontweight='bold')
                bottom += pct

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=30, ha='right', fontsize=14)
    ax.set_ylabel('Time (%)', fontsize=14)
    ax.set_title(f'Layer Time Distribution - {filename}', fontsize=16, pad=70)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Legend: one entry per base type, placed just below the title
    handles = [plt.Rectangle((0, 0), 1, 1, color=base_colors[bt]) for bt in base_types]
    ax.legend(handles, base_types, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=min(6, len(base_types)), fontsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot layer performance from output file.')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the output file (e.g. outs/OIANet_8)')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save the plot (default: plots)')
    args = parser.parse_args()

    filename = os.path.basename(args.file)

    fw, bw = parse_output_file(args.file)
    if not fw:
        print(f"No forward layer data found in {args.file}")
    else:
        plot_stacked_bar(fw, bw, filename, args.output_dir)
