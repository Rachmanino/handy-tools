import re
import matplotlib.pyplot as plt
import numpy as np

def parse_marlin_log(filepath):
    """Parses the marlin log file to extract shape and TFLOPS."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r"Shape: (\d+)x\d+x\d+,.*? (\d+\.\d+) TFLOPS", line)
            if match:
                shape = match.group(1)
                tflops = float(match.group(2))
                data[shape] = tflops
    return data

def parse_tilelang_log(filepath):
    """Parses the tile-lang log file to extract shape and TFLOPS."""
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("Shape:"):
                shape_match = re.search(r"Shape:  (\d+)", line)
                if shape_match and i + 2 < len(lines):
                    shape = shape_match.group(1)
                    tflops_line = lines[i+2]
                    tflops_match = re.search(r"Tile-lang: (\d+\.\d+) TFlops", tflops_line)
                    if tflops_match:
                        tflops = float(tflops_match.group(1))
                        data[shape] = tflops
                i += 3
            else:
                i += 1
    return data

def plot_comparison(marlin_data, tilelang_data):
    """Plots a bar chart comparing Marlin and TileLang TFLOPS."""
    labels = sorted(marlin_data.keys(), key=int)
    marlin_values = [marlin_data[label] for label in labels]
    tilelang_values = [tilelang_data.get(label, 0) for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, marlin_values, width, label='Marlin')
    rects2 = ax.bar(x + width/2, tilelang_values, width, label='TileLang')

    ax.set_ylabel('TFLOPS')
    ax.set_title('BF16MXFP4 GEMM Performance Comparison, N=K=8192')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={label}' for label in labels], fontsize=14)
    ax.legend(fontsize=16)  # 放大图例

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig("tflops_comparison.png")
    print("Plot saved to tflops_comparison.png")

if __name__ == "__main__":
    marlin_log_file = 'tilescale/test_marlin.log'
    tilelang_log_file = 'tilescale/test_mxfp4_tilelang.txt'

    marlin_data = parse_marlin_log(marlin_log_file)
    tilelang_data = parse_tilelang_log(tilelang_log_file)

    plot_comparison(marlin_data, tilelang_data)
