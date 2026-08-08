"""Matplotlib plots of the element geometry colored by a response quantity.

The element is drawn as a horizontal band whose height is the (exaggerated,
not-to-scale) tube diameter, and the chosen quantity is mapped onto a
centerline color gradient with a colorbar. Supports are marked and the
deflected shape can be overlaid with a stated magnification factor.
"""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from elementload.geometry import tube_section_modulus

QUANTITIES = ("stress", "moment", "load", "wind_load", "pressure", "deflection")

_LABELS = {
    "stress": "bending stress (MPa)",
    "moment": "bending moment (N.m)",
    "load": "distributed load (N/m)",
    "wind_load": "wind load (N/m)",
    "pressure": "wind pressure (Pa)",
    "deflection": "deflection (mm)",
}


def _hermite_deflection(u, L, s):
    """Cubic Hermite transverse deflection at local coordinate s in [0, L].
    u = (w_i, theta_i, w_j, theta_j)."""
    w_i, t_i, w_j, t_j = u
    xi = s / L
    return (2 * xi**3 - 3 * xi**2 + 1) * w_i \
        + L * (xi**3 - 2 * xi**2 + xi) * t_i \
        + (-2 * xi**3 + 3 * xi**2) * w_j \
        + L * (xi**3 - xi**2) * t_j


def _element_u(result, e, nodes):
    i, j = e.i, e.j
    return (result.displacements[i][0], result.displacements[i][1],
            result.displacements[j][0], result.displacements[j][1])


def _sample_quantity(model, result, quantity, per_element=6):
    """Sample (x, value) along every element for the given quantity."""
    pts = []
    for i, seg in enumerate(model.segments):
        L = seg.x1 - seg.x0
        for k in range(per_element + 1):
            s = L * k / per_element
            x = seg.x0 + s
            if quantity == "stress":
                from elementload.solver import internal_moment
                m = internal_moment(model.elements, i, model.nodes,
                                    result.displacements, model.loads, s)
                z = tube_section_modulus(seg.outer_d, seg.wall)
                val = abs(m) / z
            elif quantity == "moment":
                from elementload.solver import internal_moment
                val = abs(internal_moment(model.elements, i, model.nodes,
                                          result.displacements, model.loads, s))
            elif quantity == "load":
                val = model.loads[i].q
            elif quantity == "wind_load":
                from elementload.report import wind_load
                val = wind_load(model.wind_speed, seg.outer_d)
            elif quantity == "pressure":
                from elementload.report import wind_load
                val = wind_load(model.wind_speed, seg.outer_d) / seg.outer_d
            elif quantity == "deflection":
                e = model.elements[i]
                u = _element_u(result, e, model.nodes)
                val = abs(_hermite_deflection(u, L, s))
            else:
                raise ValueError(f"unknown quantity: {quantity}")
            pts.append((x, val))
    return pts


def plot_element(model, result, path, quantity="stress", title=None,
                 deformed=False):
    """Render the element colored by `quantity` and save to `path`.

    Geometry is not to scale: the tube diameters are exaggerated so the stepped
    telescoping profile is visible next to the element length. If `deformed` is
    true the deflected shape is overlaid with a labelled magnification factor.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity: {quantity}")
    pts = _sample_quantity(model, result, quantity)
    xs = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    span = xmax - xmin or 1.0

    fig, ax = plt.subplots(figsize=(12, 4))

    # element geometry band (exaggerated diameter, not to scale)
    max_d = max(seg.outer_d for seg in model.segments)
    d_scale = max(span / 25, 1e-6) / max_d  # largest tube ~4% of span
    d_note = f"diameters exaggerated x{d_scale:.0f} (not to scale)"
    for seg in model.segments:
        d = seg.outer_d * d_scale / 2
        ax.fill_between([seg.x0, seg.x1], -d, d, color="#d9d9d9", linewidth=0)
    ax.axhline(0, color="#b0b0b0", linewidth=0.8)

    # colored centerline via LineCollection
    norm = Normalize(min(vals) if min(vals) != max(vals) else 0.0,
                     max(vals) if max(vals) != min(vals) else 1.0)
    segs = [[(xs[k], 0.0), (xs[k + 1], 0.0)] for k in range(len(xs) - 1)]
    lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=6,
                        capstyle="round")
    lc.set_array(vals[:-1])
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label=_LABELS[quantity], fraction=0.04, pad=0.02)

    # supports
    for sup in model.supports:
        x = model.nodes[sup.node].x
        y = max(seg.outer_d for seg in model.segments) * d_scale / 2
        if sup.kind == "fixed":
            ax.plot(x, y, marker="o", color="k", markersize=8)
        else:
            ax.plot(x, y, marker="^", color="k", markersize=8)

    # deformed shape overlay
    if deformed:
        u_pts = _sample_quantity(model, result, "deflection")
        max_w = max(v for _, v in u_pts) or 1.0
        mag = max(span / 8, 1e-6) / max_w
        dx = [p[0] for p in u_pts]
        dy = [p[1] * mag for p in u_pts]
        ax.plot(dx, dy, color="k", linewidth=1.2, linestyle="--", alpha=0.8,
                label=f"deformed shape (x{mag:.0f})")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("position along element (m)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-span / 40, span / 40)
    ax.tick_params(axis="y", labelleft=False)
    ax.text(0.01, 0.95, d_note, transform=ax.transAxes, fontsize=8,
            color="0.3", verticalalignment="top")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
