import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import arviz as az
import qmricolors
import matplotlib.patheffects as pe

PLOT_SETTINGS = {"text.usetex": True,
                 "font.family": "serif" ,
                 'text.latex.preamble': r"\usepackage{bm}",
                 }
plt.rcParams.update(PLOT_SETTINGS)
# -------------------------------
# Load T1 ML estimate and masks
# -------------------------------
import pickle
from utils import expand_mask, load_data

rs = 256
T1_gt, M_gt, mask, A1,A2,A3, y_obs = load_data(rs)
mask = mask.astype(bool)
mask_arr = mask != 0
gt_flaten = T1_gt[mask_arr]
T1_gt = expand_mask(gt_flaten,mask)


# Define tissue masks (already cleaned)
wm_mask = (T1_gt > 0.8) & (T1_gt < 1.1)
gm_mask = (T1_gt >= 1.1) & (T1_gt < 1.6)
csf_mask = (T1_gt > 2.0) & (T1_gt < 5.0)

from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.morphology import remove_small_objects, remove_small_holes

# Erode CSF to avoid overlap
csf_mask = binary_erosion(csf_mask, iterations=1)
csf_center = np.zeros_like(csf_mask, dtype=bool)
csf_center[128:150,100:150] = True
# csf_center[100:106, 100:106] = True
csf_mask[~csf_center] = 0

# Exclude CSF from WM/GM
wm_mask = wm_mask & ~csf_mask
gm_mask = gm_mask & ~csf_mask

def clean_mask(mask, area_threshold=64, min_size=64):
    mask_clean = binary_fill_holes(mask)
    mask_clean = mask
    mask_clean = remove_small_holes(mask_clean, area_threshold=area_threshold)
    mask_clean = remove_small_objects(mask_clean, min_size=min_size)
    mask_clean[:,128:] = 0
    mask_clean[:128,:] = 0
    return mask_clean

wm_mask_clean = clean_mask(wm_mask)
gm_mask_clean = clean_mask(gm_mask)
gm_mask_clean = gm_mask_clean & ~wm_mask_clean


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def indexs(mask_new, mask=mask):
    mask_flat = mask.flatten()  # original mask used to generate uni.values
    mask_new_flat = mask_new.flatten()  # 256x256 mask

    # Restrict mask_new to the original acquisition mask
    mask_in_acq = mask_new_flat[mask_flat]  # boolean array of size n_masked_voxels
    tissue_indices = np.where(mask_in_acq)[0]  # indices relative to T1_samples

    return tissue_indices

# -------------------------------
# Function to plot KDE PDF
# -------------------------------
def plot_pdf(data, label, ax, color, linestyle='-'):
    """Plot KDE PDF of the data and mark the mode."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return
    kernel = stats.gaussian_kde(data)
    opt = optimize.minimize_scalar(lambda x: -kernel(x))  # mode
    xs = np.linspace(0, data.max(), 201)
    ax.plot(xs, kernel(xs), color=color, linestyle=linestyle, label=label)
    ax.plot(opt.x, kernel(opt.x), '.', color=color)
    mean = np.mean(data)
    std = 2*np.std(data)
    print(label, f'{opt.x.item():.2f},', f'{mean:.2f} $\pm$ {std:.2f}')

# -------------------------------
# Load posterior samples from ArviZ idata
# -------------------------------
idata_uni = az.from_netcdf("results/uni.nc")
idata_gamma = az.from_netcdf("results/gamma.nc")
idata_tvuni = az.from_netcdf("results/tvuni.nc")
idata_tvl1 = az.from_netcdf("results/tvl1.nc")
idata_tvl1prior = az.from_netcdf("results/Htvl1.nc")

uni = az.extract(idata_uni, var_names="T1")
gamma = az.extract(idata_gamma, var_names="T1")
tvuni = az.extract(idata_tvuni, var_names="T1")
tvl1 = az.extract(idata_tvl1, var_names="T1")
tvl1prior = az.extract(idata_tvl1prior, var_names="T1")


with open('results/map_estimates.pkl', 'rb') as f:
    data = pickle.load(f)
T1_ML = expand_mask(data['T1'], mask)
ML_flaten = T1_ML[mask_arr]

plt.imshow(T1_ML,cmap='lipari', vmin=0, vmax=4)
plt.show()
# -------------------------------
# Prepare tissue-masked values for each method
# -------------------------------


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# -------------------------------
# Plot PDFs: one subplot per tissue
# -------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.99)

# Function to create a colormap that shows 0 as white
def mask_overlay(mask, color):
    # Convert to float
    mask_float = mask.astype(float)
    # Mask the zeros so they are not plotted
    masked = np.ma.masked_where(mask_float == 0, mask_float)
    return masked

ax.imshow(T1_gt.T, cmap='lipari')
ax.axis("off")
zoom_xmin, zoom_xmax = 40, 220 # x-axis limits for zooming
zoom_ymin, zoom_ymax = 230, 20    # y-axis limits for zooming
ax.set_xlim(zoom_xmin, zoom_xmax)
ax.set_ylim(zoom_ymin, zoom_ymax)

# Overlay masks with colors
ax.imshow(mask_overlay(wm_mask_clean.T, 'red'), cmap=ListedColormap(['white','red']),
           alpha=0.8, origin='lower', vmin=0, vmax=1)
ax.imshow(mask_overlay(gm_mask_clean.T, 'blue'), cmap=ListedColormap(['white','blue']),
           alpha=0.8, origin='lower', vmin=0, vmax=1)
ax.imshow(mask_overlay(csf_mask.T, 'green'), cmap=ListedColormap(['white','green']),
           alpha=0.8, origin='lower', vmin=0, vmax=1)

# Example manual coordinates (x, y)
red_coord   = (155, 91)  # for WM
blue_coord  = (165, 62)   # for GM
green_coord = (144, 145)  # for CSF

# Add numbers with circles
ax.text(
    red_coord[0], red_coord[1], '1',
    color='white',
    fontsize=33,
    fontweight='bold',
    ha='center', va='center',
    path_effects=[pe.withStroke(linewidth=4, foreground='black')]
)

ax.text(
    blue_coord[0], blue_coord[1], '2',
    color='white',
    fontsize=33,
    fontweight='bold',
    ha='center', va='center',
    path_effects=[pe.withStroke(linewidth=4, foreground='black')]
)

ax.text(
    green_coord[0], green_coord[1], '3',
    color='white',
    fontsize=33,
    fontweight='bold',
    ha='center', va='center',
    path_effects=[pe.withStroke(linewidth=4, foreground='black')]
)

plt.tight_layout()
plt.savefig('pdfs_brain_1.pdf', dpi=300)


methods = {
    r"$U(0,50)$": uni.values,
    r"Gamma(3,1)": gamma.values,
    r"Bounded TV": tvuni.values,
    r"Hierarch.~TV$^1_\mu$": tvl1prior.values,
    r"TV$^1_\mu$": tvl1.values,
}

tissue_masks = {
    r'\textbf{(1) WM}': indexs(wm_mask_clean),
    r'\textbf{(2) GM}': indexs(gm_mask_clean),
    r'\textbf{(3) CSF}': indexs(csf_mask)
}

tissue_colors = {
    r'\textbf{(1) WM}': 'red',
    r'\textbf{(2) GM}': 'blue',
    r'\textbf{(3) CSF}': 'green'
}
# Define colors and line styles for each method
method_colors = {
    r"$U(0,50)$": 'orange',
    r"Gamma(3,1)": 'b',
    r"Bounded TV": 'green',
    r"Hierarch.~TV$^1_\mu$": "k",
    r"TV$^1_\mu$": 'k',
}

linestyles = ['-', '-', '-', '--', '-']  # optional




# def gt_ml(gt_data,ml_data):
#     gt_kde = stats.gaussian_kde(gt_data)
#     ml_kde = stats.gaussian_kde(ml_data)

#     # Create x values
#     x_vals = np.linspace(min(gt_data.min(), ml_data.min()),
#                         max(gt_data.max(), ml_data.max()), 500)

#     # Plot PDFs
#     ax.plot(x_vals, gt_kde(x_vals), color='m', label='GT', linewidth=2)
#     ax.plot(x_vals, ml_kde(x_vals), color='m', linestyle='--', label='MLE', linewidth=2)


for i, (tissue, mask_vals) in enumerate(tissue_masks.items()):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.99)

    gt_mean = gt_flaten[mask_vals].mean()
    ml_mean = ML_flaten[mask_vals].mean()
    print(gt_mean,ml_mean)
    ax.axvline(gt_mean, color='m', label="GT")
    ax.axvline(ml_mean, color='m', linestyle='--', label="MLE")

    if tissue == r'\textbf{(1) WM}':
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0.0, 5)
        ax.set_ylabel("Probability Density", fontsize=28)
    elif tissue == r'\textbf{(2) GM}':
        ax.set_xlim(0.5, 2.5)
    elif tissue == r'\textbf{(3) CSF}':
        ax.set_xlim(1, 6)

    for j, (method_name, T1_samples) in enumerate(methods.items()):
        data_vals = T1_samples[mask_vals, :].flatten()
        plot_pdf(data_vals, method_name, ax, color=method_colors[method_name],
                linestyle=linestyles[j % len(linestyles)])
        
    ax.tick_params(axis='both', labelsize=26)
    ax.set_title(tissue, fontsize=32)
    ax.set_xlabel(r"$T_1$ (s)", fontsize=28)

    plt.tight_layout()
    plt.savefig(f'pdfs_brain_{i+3}.pdf', dpi=300)

    if i==0:
        handles, labels = ax.get_legend_handles_labels()
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.99)

        ax.axis("off")
        ax.legend(handles, labels, ncol=1, loc="center", title=r"\textbf{Methods}",
                 title_fontsize=30, fontsize=28)
        
        plt.tight_layout()
        plt.savefig('pdfs_brain_2.pdf', dpi=300)
        plt.show()

