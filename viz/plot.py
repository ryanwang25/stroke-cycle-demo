import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_boundaries(frames, right_wrist_x, right_shoulder_x, boundaries, show_scatter=False):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(frames, right_wrist_x, color='orange', label='R wrist x', linewidth=1.5)
    ax.plot(frames, right_shoulder_x, color='purple', label='R shoulder x', linewidth=1.5, linestyle='--')

    if show_scatter:
        ax.scatter(frames, right_wrist_x, color='orange', s=10, alpha=0.5)
        ax.scatter(frames, right_shoulder_x, color='purple', s=10, alpha=0.5)

    for b in boundaries:
        ax.axvline(x=b, color='green', linewidth=1.5, alpha=0.7)
    green_patch = mpatches.Patch(color='green', label='boundary detected')
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], green_patch])
    ax.set_xlabel("frame")
    ax.set_ylabel("normalized x position")
    ax.set_title("Stroke Cycle Boundary Detection — Right Arm")
    plt.tight_layout()
    plt.savefig("output/boundary_plot.png")
    print("Plot saved to output/boundary_plot.png")