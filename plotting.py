from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from nilearn import datasets
from nilearn.surface import load_surf_data
from nilearn.plotting.surface._matplotlib_backend import _plot_surf, _colorbar_from_array, _get_ticks
from mreyemove.plotting.glm import custom_cmap
import numpy as np


def plot_surface_searchlight(
        corr_maps,
        surf_mesh="fsaverage7",
        cmap=None,
        vmin=-1,
        vmax=1,
        threshold=None,
        symmetric_cbar="auto",
        cbar_tick_format="%.1f",
        title=None,
        inflate=False,
        output_file=None,
        bg_on_data=True,
        mask=None,
):
    """
    Plot surface searchlight correlation maps using the same rendering
    pipeline as glm_surface.py (internal _plot_surf).
    """
    if cmap is None:
        cmap = custom_cmap()

    fsaverage = datasets.fetch_surf_fsaverage(mesh=surf_mesh)
    hemis = ["left", "right"]
    modes = ["lateral", "medial"]

    if symmetric_cbar is None:
        symmetric_cbar = "auto"
    if cbar_tick_format is None:
        cbar_tick_format = "%i"

    cbar_h = 0.25
    title_h = 0.25 * (title is not None)
    w, h = plt.figaspect((1 + cbar_h + title_h) / (len(hemis) + len(modes)))
    fig = plt.figure(figsize=(w * 2, h * 2), constrained_layout=False)
    height_ratios = [title_h] + [1.0] + [cbar_h]
    grid = GridSpec(
        nrows=3,
        ncols=len(modes) + len(hemis),
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.0,
    )
    axes = []

    panels = [
        ("left", "lateral"),
        ("right", "lateral"),
        ("right", "medial"),
        ("left", "medial"),
    ]
    mesh_prefix = "infl" if inflate else "pial"
    surf = {
        "left": fsaverage[f"{mesh_prefix}_left"],
        "right": fsaverage[f"{mesh_prefix}_right"],
    }

    for i, (hemi, mode) in enumerate(panels):
        if inflate:
            curv_map = load_surf_data(fsaverage[f"curv_{hemi}"])
            curv_sign_map = (np.sign(curv_map) + 1) / 4 + 0.25
            bg_map = curv_sign_map
        else:
            sulc_map = fsaverage[f"sulc_{hemi}"]
            bg_map = sulc_map

        grid_idx = i + (len(hemis) + len(modes))

        ax = fig.add_subplot(grid[grid_idx], projection="3d")
        axes.append(ax)

        surf_map = corr_maps[hemi].copy()

        if mask is not None:
            # Mask everywhere where the mask is False (with nan)
            surf_map[~mask[hemi]] = np.nan

        _plot_surf(
            surf_mesh=surf[hemi],
            surf_map=surf_map,
            bg_map=bg_map,
            hemi=hemi,
            view=mode,
            cmap=cmap,
            darkness=None,
            colorbar=False,
            threshold=threshold,
            bg_on_data=bg_on_data,
            vmin=vmin,
            vmax=vmax,
            axes=ax,
        )
        ax.set_box_aspect(None, zoom=1.3)

    # Colorbar
    sm = _colorbar_from_array(
        np.concatenate([corr_maps["left"], corr_maps["right"]]),
        vmin, vmax, threshold,
        symmetric_cbar=symmetric_cbar,
        cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
    )
    cbar_grid = GridSpecFromSubplotSpec(3, 3, grid[-1, :])
    cbar_ax = fig.add_subplot(cbar_grid[1])
    ticks = _get_ticks(vmin, vmax, cbar_tick_format, threshold)
    fig.colorbar(
        sm, cax=cbar_ax, orientation="horizontal",
        ticks=ticks, format=cbar_tick_format,
    )

    if title is not None:
        fig.suptitle(title, y=1.0 - title_h / sum(height_ratios), va="bottom")

    if output_file:
        fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig