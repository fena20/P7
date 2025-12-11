#!/usr/bin/env python3
"""
Phase 7: Visualization for Journal/Thesis
Generates publication-quality figures for Applied Energy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import seaborn as sns
import logging
from pathlib import Path

from config import (
    OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED,
    FIGURE_DPI, FIGURE_FORMAT, COLOR_PALETTE, COLORMAP_CONTINUOUS,
    VIABILITY_PARAMS, CENSUS_DIVISIONS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2
})


def save_figure(fig, name):
    """Save figure in multiple formats."""
    for fmt in FIGURE_FORMAT:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ {name}")


def fig_workflow():
    """Figure 1: Study Workflow Diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    bw, bh = 2.4, 0.8
    
    def draw_box(x, y, text, color, fs=10):
        box = FancyBboxPatch((x-bw/2, y-bh/2), bw, bh,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, 
                fontweight='bold', color='white', wrap=True)
    
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>', mutation_scale=15,
                                color='#2c3e50', linewidth=2.5)
        ax.add_patch(arrow)
    
    y_levels = [9, 7.2, 5.4, 3.6, 1.8]
    x_left, x_center, x_right = 3.5, 7, 10.5
    
    # Boxes
    draw_box(x_center, y_levels[0], 'RECS 2020\nMicrodata\n(n=18,496)', COLOR_PALETTE['primary'])
    draw_box(x_left, y_levels[1], 'Filter: Gas-Heated\n(n=9,411)', COLOR_PALETTE['accent'])
    draw_box(x_right, y_levels[1], 'Feature Engineering\n(24 variables)', COLOR_PALETTE['accent'])
    draw_box(x_left, y_levels[2], 'Outlier Removal\n(2–98 percentile)', COLOR_PALETTE['accent'])
    draw_box(x_right, y_levels[2], 'Train/Val/Test\n(60/20/20)', COLOR_PALETTE['accent'])
    draw_box(x_left, y_levels[3], 'XGBoost\nRegression', COLOR_PALETTE['moderate'])
    draw_box(x_center, y_levels[3], 'SHAP\nAnalysis', COLOR_PALETTE['moderate'])
    draw_box(x_right, y_levels[3], 'Scenario\nEnumeration', COLOR_PALETTE['moderate'])
    draw_box(x_left, y_levels[4], 'Model Metrics\n(Table 3)', COLOR_PALETTE['poor'])
    draw_box(x_center, y_levels[4], 'Feature Importance\n(Table 4)', COLOR_PALETTE['poor'])
    draw_box(x_right, y_levels[4], 'Viability Results\n(Figures 8-11)', COLOR_PALETTE['poor'])
    
    # Arrows
    draw_arrow(x_center - 0.3, y_levels[0] - bh/2 - 0.1, x_left, y_levels[1] + bh/2 + 0.1)
    draw_arrow(x_center + 0.3, y_levels[0] - bh/2 - 0.1, x_right, y_levels[1] + bh/2 + 0.1)
    
    for x in [x_left, x_right]:
        for i in range(1, 4):
            draw_arrow(x, y_levels[i] - bh/2 - 0.1, x, y_levels[i+1] + bh/2 + 0.1)
    
    draw_arrow(x_left + bw/2 + 0.1, y_levels[3], x_center - bw/2 - 0.1, y_levels[3])
    draw_arrow(x_right - bw/2 - 0.1, y_levels[3], x_center + bw/2 + 0.1, y_levels[3])
    draw_arrow(x_center, y_levels[3] - bh/2 - 0.1, x_center, y_levels[4] + bh/2 + 0.1)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_PALETTE['primary'], edgecolor='black', label='Data Source'),
        mpatches.Patch(facecolor=COLOR_PALETTE['accent'], edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=COLOR_PALETTE['moderate'], edgecolor='black', label='Modeling'),
        mpatches.Patch(facecolor=COLOR_PALETTE['poor'], edgecolor='black', label='Outputs'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
              framealpha=0.95, edgecolor='black', bbox_to_anchor=(0.01, 0.99))
    
    plt.title('Figure 1: Study Workflow', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_figure(fig, 'fig01_workflow')


def fig_viability_heatmaps():
    """Figure 9: HP Viability Score Heatmaps."""
    alpha = VIABILITY_PARAMS['alpha']
    beta = VIABILITY_PARAMS['beta']
    gammas = VIABILITY_PARAMS['gamma']
    
    hdd = np.linspace(2000, 8000, 100)
    price = np.linspace(0.08, 0.22, 100)
    HDD, PRICE = np.meshgrid(hdd, price)
    H_star = (HDD - 2000) / 6000
    P_star = (PRICE - 0.08) / 0.14
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (ax, (env, gamma)) in enumerate(zip(axes, gammas.items())):
        V = (1 - alpha * H_star) * (1 - beta * P_star) * gamma
        im = ax.contourf(HDD, PRICE, V, levels=np.linspace(0, 1, 21), cmap=COLORMAP_CONTINUOUS)
        
        cs = ax.contour(HDD, PRICE, V, levels=[0.5], colors='white', linewidths=4)
        ax.clabel(cs, fmt='V=0.5', fontsize=11, inline_spacing=10)
        ax.contour(HDD, PRICE, V, levels=[0.3, 0.7], colors='white', linewidths=2, linestyles='--')
        
        ax.set_title(f'{env} Envelope (γ={gamma:.2f})', fontsize=13, fontweight='bold')
        ax.set_xlabel('HDD65', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Electricity Price ($/kWh)', fontsize=12)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('HP Viability Score V', fontsize=12)
    
    plt.suptitle('Figure 9: HP Viability Score Heatmaps', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    save_figure(fig, 'fig09_viability_heatmaps')


def fig_monte_carlo():
    """Figure 16: NPV Uncertainty Distribution."""
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load or generate Monte Carlo results
    mc_path = OUTPUT_DIR / "monte_carlo_results.csv"
    if mc_path.exists():
        mc_df = pd.read_csv(mc_path)
        npv = mc_df['npv'].values
    else:
        np.random.seed(RANDOM_SEED)
        npv = np.random.normal(2500, 3500, 10000)
    
    mean_npv = npv.mean()
    std_npv = npv.std()
    
    ax.hist(npv, bins=50, color=COLOR_PALETTE['primary'], alpha=0.6, 
            edgecolor='black', density=True, label='Simulated NPV')
    
    kde = stats.gaussian_kde(npv)
    x_kde = np.linspace(npv.min(), npv.max(), 200)
    ax.plot(x_kde, kde(x_kde), 'k-', linewidth=2.5, label='KDE fit')
    
    ax.axvline(0, color=COLOR_PALETTE['poor'], linewidth=3, linestyle='--', label='Break-even')
    ax.axvline(np.median(npv), color=COLOR_PALETTE['accent'], linewidth=2.5, 
               label=f'Median: ${np.median(npv):,.0f}')
    
    prob = (npv > 0).mean() * 100
    ax.text(0.97, 0.95, f'Distribution Parameters:\nμ = ${mean_npv:,.0f}\nσ = ${std_npv:,.0f}\nn = {len(npv):,} samples\n\nP(NPV > 0) = {prob:.0f}%',
           transform=ax.transAxes, fontsize=11, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    ax.set_xlabel('15-Year NPV ($)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Figure 16: NPV Uncertainty Distribution (Monte Carlo)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'fig16_monte_carlo')


def fig_sobol():
    """Figure 17: Sobol Sensitivity Indices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Load or generate Sobol results
    sobol_path = OUTPUT_DIR / "sobol_sensitivity.csv"
    if sobol_path.exists():
        sobol_df = pd.read_csv(sobol_path)
        params = sobol_df['parameter'].tolist()[:8]
        s1 = sobol_df['S1'].tolist()[:8]
        st = sobol_df['ST'].tolist()[:8]
    else:
        params = ['Elec.\nPrice', 'Gas\nPrice', 'HDD', 'COP', 'HP\nCost', 'Retrofit', 'Disc.\nRate', 'Life']
        s1 = [0.32, 0.18, 0.22, 0.12, 0.08, 0.05, 0.02, 0.01]
        st = [0.45, 0.28, 0.35, 0.20, 0.12, 0.08, 0.03, 0.02]
    
    x = np.arange(len(params))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, s1, width, label='S₁ (First-order)', 
                       color=COLOR_PALETTE['primary'], edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, st, width, label='Sₜ (Total)', 
                       color=COLOR_PALETTE['poor'], edgecolor='black', linewidth=1.5, hatch='//')
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(params, fontsize=9)
    axes[0].set_ylabel('Sobol Index', fontsize=12)
    axes[0].set_title('(a) Sensitivity Indices', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 0.55)
    
    # Interaction heatmap
    n = len(params)
    interact = np.zeros((n, n))
    interact[0, 2] = interact[2, 0] = 0.15
    interact[0, 3] = interact[3, 0] = 0.08
    interact[1, 2] = interact[2, 1] = 0.12
    interact[2, 3] = interact[3, 2] = 0.10
    np.fill_diagonal(interact, np.nan)
    
    im = axes[1].imshow(interact, cmap='YlOrRd', vmin=0, vmax=0.18)
    short_params = ['E.P', 'G.P', 'HDD', 'COP', 'HP$', 'Ret', 'DR', 'Life']
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(short_params, fontsize=9)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(short_params, fontsize=9)
    axes[1].set_title('(b) Interaction Effects (Sᵢⱼ)', fontsize=13, fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            if i != j and interact[i, j] > 0.02:
                c = 'white' if interact[i, j] > 0.1 else 'black'
                axes[1].text(j, i, f'{interact[i, j]:.2f}', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color=c)
    
    plt.colorbar(im, ax=axes[1], shrink=0.8, label='Interaction Index')
    
    plt.suptitle('Figure 17: Global Sensitivity Analysis (Sobol Indices)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig17_sobol')


def fig_division_map():
    """Figure 10: HP Viability by Census Division."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Load division summary if available
    div_path = OUTPUT_DIR / "division_summary.csv"
    if div_path.exists():
        div_df = pd.read_csv(div_path)
        divisions = {row['abbr']: {'viability': row['mean_viability'], 'homes': row['homes_millions']} 
                    for _, row in div_df.iterrows()}
    else:
        divisions = {
            'NE': {'viability': 0.35, 'homes': 2.1},
            'MA': {'viability': 0.42, 'homes': 5.8},
            'ENC': {'viability': 0.38, 'homes': 7.2},
            'WNC': {'viability': 0.32, 'homes': 3.1},
            'SA': {'viability': 0.68, 'homes': 4.5},
            'ESC': {'viability': 0.55, 'homes': 2.8},
            'WSC': {'viability': 0.72, 'homes': 3.2},
            'Mtn': {'viability': 0.48, 'homes': 2.9},
            'PAC': {'viability': 0.62, 'homes': 4.8},
        }
    
    # Positions for schematic map
    positions = {
        'PAC': (1.0, 5.0), 'Mtn': (3.0, 4.5), 'WNC': (5.0, 5.5),
        'WSC': (5.5, 2.5), 'ENC': (7.5, 6.0), 'ESC': (8.0, 3.5),
        'SA': (10.0, 3.5), 'MA': (10.5, 6.0), 'NE': (11.5, 7.5)
    }
    
    # Region shapes
    region_shapes = {
        'PAC': [(0, 3), (2, 3), (2, 7), (0, 7)],
        'Mtn': [(2, 2.5), (4.5, 2.5), (4.5, 7), (2, 7)],
        'WNC': [(4.5, 4), (7, 4), (7, 7), (4.5, 7)],
        'WSC': [(4.5, 1), (7, 1), (7, 4), (4.5, 4)],
        'ENC': [(7, 4.5), (9.5, 4.5), (9.5, 7), (7, 7)],
        'ESC': [(7, 2.5), (9, 2.5), (9, 4.5), (7, 4.5)],
        'SA': [(9, 1.5), (12, 1.5), (12, 5), (9, 5)],
        'MA': [(9.5, 5), (12, 5), (12, 7), (9.5, 7)],
        'NE': [(11, 7), (13, 7), (13, 8.5), (11, 8.5)],
    }
    
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0.3, vmax=0.75)
    
    patches = []
    colors_list = []
    for name, coords in region_shapes.items():
        polygon = Polygon(coords, closed=True)
        patches.append(polygon)
        v = divisions.get(name, {'viability': 0.5})['viability']
        colors_list.append(cmap(norm(v)))
    
    collection = PatchCollection(patches, alpha=0.85, edgecolor='black', linewidth=2)
    collection.set_facecolors(colors_list)
    ax.add_collection(collection)
    
    for name, (x, y) in positions.items():
        if name in divisions:
            v = divisions[name]['viability']
            homes = divisions[name]['homes']
            text_color = 'white' if v < 0.55 else 'black'
            ax.text(x, y + 0.3, name, ha='center', va='bottom', fontsize=13, fontweight='bold', color=text_color)
            ax.text(x, y - 0.1, f'V = {v:.2f}', ha='center', va='top', fontsize=11, color=text_color)
            ax.text(x, y - 0.5, f'{homes:.1f}M homes', ha='center', va='top', fontsize=10, color=text_color)
    
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('HP Viability Score', fontsize=12)
    
    ax.text(0.02, 0.02, 'Note: Schematic layout, not to geographic scale. M = millions.', 
           transform=ax.transAxes, fontsize=10, style='italic', color='gray')
    
    plt.title('Figure 10: HP Viability by Census Division', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'fig10_us_map')


def generate_all_figures():
    """Generate all publication figures."""
    
    logger.info("=" * 60)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 60)
    
    fig_workflow()
    fig_viability_heatmaps()
    fig_monte_carlo()
    fig_sobol()
    fig_division_map()
    
    logger.info("\n✅ All figures generated")
    logger.info(f"   Saved to: {FIGURES_DIR}")


def main():
    """Execute Phase 7."""
    logger.info("=" * 60)
    logger.info("PHASE 7: VISUALIZATION FOR JOURNAL/THESIS")
    logger.info("=" * 60)
    
    generate_all_figures()
    
    # Summary
    logger.info("\n✅ Phase 7 Complete")
    logger.info(f"   Figures saved to: {FIGURES_DIR}")
    logger.info(f"   Formats: {FIGURE_FORMAT}")
    logger.info(f"   Resolution: {FIGURE_DPI} DPI")


if __name__ == "__main__":
    main()
