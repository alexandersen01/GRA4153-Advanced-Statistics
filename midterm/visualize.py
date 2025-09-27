import matplotlib.pyplot as plt

def plot_unified_comparison(results):
    """Create a unified plot comparing all rejection rates and estimators"""
    df = pd.DataFrame(results)
    df["p_diff"] = np.abs(df["p1"] - df["p2"])

    # Create a single large plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define colors and markers for different combinations
    sample_size_configs = df[["n1", "n2"]].drop_duplicates().values
    test_stats = ["rej_Z1", "rej_Z2", "rej_Z3"]
    test_names = ["Z1 (Pooled)", "Z2 (Unpooled)", "Z3 (Unpooled)"]

    # Color scheme: different colors for sample sizes, different line styles for test statistics
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]

    # Plot each combination
    for i, (n1, n2) in enumerate(sample_size_configs):
        subset = df[(df["n1"] == n1) & (df["n2"] == n2)]
        subset_sorted = subset.sort_values("p_diff")

        for j, (test_stat, test_name) in enumerate(zip(test_stats, test_names)):
            label = f"{test_name} (n1={n1}, n2={n2})"
            ax.plot(
                subset_sorted["p_diff"],
                subset_sorted[test_stat],
                color=colors[i],
                linestyle=linestyles[j],
                label=label,
                linewidth=3,
                alpha=0.85,
            )

    ax.set_xlabel("|p1 - p2|", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rejection Rate", fontsize=14, fontweight="bold")
    ax.set_title(
        "Unified Comparison: Rejection Rates vs |p1 - p2|\nfor All Test Statistics and Sample Sizes",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.01, 0.21)

    # Add horizontal line at alpha level
    ax.axhline(
        y=0.05, color="red", linestyle=":", alpha=0.7, linewidth=2, label="α = 0.05"
    )

    plt.tight_layout()
    plt.savefig("unified_rejection_rates_comparison.png", dpi=300, bbox_inches="tight")

    
    plt.show()


def plot_focused_comparison(results):
    """Create a focused comparison plot with clearer visualization"""
    df = pd.DataFrame(results)
    df["p_diff"] = np.abs(df["p1"] - df["p2"])

    # Create subplots: one for each sample size configuration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Test Statistics Comparison by Sample Size Configuration",
        fontsize=16,
        fontweight="bold",
    )

    sample_size_configs = df[["n1", "n2"]].drop_duplicates().values
    test_stats = ["rej_Z1", "rej_Z2", "rej_Z3"]
    test_names = ["Z1 (Pooled)", "Z2 (Unpooled)", "Z3 (Unpooled)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for i, (n1, n2) in enumerate(sample_size_configs):
        ax = axes[i]
        subset = df[(df["n1"] == n1) & (df["n2"] == n2)]
        subset_sorted = subset.sort_values("p_diff")

        for j, (test_stat, test_name, color, marker) in enumerate(
            zip(test_stats, test_names, colors, markers)
        ):
            ax.plot(
                subset_sorted["p_diff"],
                subset_sorted[test_stat],
                color=color,
                label=test_name,
                linewidth=3,
                alpha=0.85,
            )

        ax.set_xlabel("|p1 - p2|", fontsize=12, fontweight="bold")
        ax.set_ylabel("Rejection Rate", fontsize=12, fontweight="bold")
        ax.set_title(f"n1={n1}, n2={n2}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.01, 0.21)

        # Add horizontal line at alpha level
        ax.axhline(y=0.05, color="red", linestyle=":", alpha=0.7, linewidth=2)

        # Add text annotation for alpha
        if i == 0:
            ax.text(0.15, 0.08, "α = 0.05", color="red", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("focused_rejection_rates_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_comparison_by_sample_size(results):
    """Create separate plots for each sample size combination"""
    df = pd.DataFrame(results)
    df["p_diff"] = np.abs(df["p1"] - df["p2"])

    # Get unique sample size combinations
    sample_sizes = df[["n1", "n2"]].drop_duplicates()

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6))
    if len(sample_sizes) == 1:
        axes = [axes]

    fig.suptitle("Rejection Rates vs |p1 - p2| by Sample Size", fontsize=16)

    colors = ["blue", "red", "green"]
    test_stats = ["rej_Z1", "rej_Z2", "rej_Z3"]
    test_names = ["Z1 (Pooled)", "Z2 (Unpooled)", "Z3 (Unpooled)"]

    for i, (_, sample_size_row) in enumerate(sample_sizes.iterrows()):
        n1, n2 = sample_size_row["n1"], sample_size_row["n2"]
        subset = df[(df["n1"] == n1) & (df["n2"] == n2)]
        subset_sorted = subset.sort_values("p_diff")

        ax = axes[i]

        for j, (test_stat, test_name, color) in enumerate(
            zip(test_stats, test_names, colors)
        ):
            ax.plot(
                subset_sorted["p_diff"],
                subset_sorted[test_stat],
                marker="o",
                color=color,
                label=test_name,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("|p1 - p2|", fontsize=12)
        ax.set_ylabel("Rejection Rate", fontsize=12)
        ax.set_title(f"n1={n1}, n2={n2}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("rejection_rates_by_sample_size.png", dpi=300, bbox_inches="tight")
    plt.show()


print("\nGenerating plots...")

# Create visualizations
plot_unified_comparison(res)
plot_focused_comparison(res)

print(
    "Plots saved as 'unified_rejection_rates_comparison.png' and 'focused_rejection_rates_comparison.png'"
)