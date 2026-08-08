"""Matplotlib plots of the element geometry colored by a response quantity.

The element is drawn as a horizontal band whose height is the (exaggerated,
not-to-scale) tube diameter, and the chosen quantity is mapped onto a
centerline color gradient with a colorbar. Supports are marked and the
deflected shape can be overlaid with a stated magnification factor.

When the element has an ES support, `plot_es_pair` draws the rigid-pin and
sag-rope models side by side so both support assumptions can be compared.
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

# factor to convert the raw SI sample to the display unit shown on the colorbar
_DISPLAY = {
    "stress": 1e-6,      # Pa -> MPa
    "moment": 1.0,       # N.m
    "load": 1.0,         # N/m
    "wind_load": 1.0,    # N/m
    "pressure": 1.0,     # Pa
    "deflection": 1e3,   # m -> mm
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
    """Sample (x, value) along every element for the given quantity. Values are
    returned in raw SI (Pa for stress, m for deflection, ...); convert with
    `_DISPLAY` before plotting."""
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


def _draw_axis(ax, model, result, quantity, norm, deformed=True):
    """Draw one element panel on `ax`. Returns the drawn LineCollection."""
    pts = _sample_quantity(model, result, quantity)
    xs = [p[0] for p in pts]
    vals = [p[1] * _DISPLAY[quantity] for p in pts]
    xmin, xmax = min(xs), max(xs)
    span = xmax - xmin or 1.0

    # element geometry band (exaggerated diameter, not to scale)
    max_d = max(seg.outer_d for seg in model.segments)
    d_scale = max(span / 25, 1e-6) / max_d  # largest tube ~4% of span
    for seg in model.segments:
        d = seg.outer_d * d_scale / 2
        ax.fill_between([seg.x0, seg.x1], -d, d, color="#d9d9d9", linewidth=0)
    ax.axhline(0, color="#b0b0b0", linewidth=0.8)

    # colored centerline via LineCollection
    segs = [[(xs[k], 0.0), (xs[k + 1], 0.0)] for k in range(len(xs) - 1)]
    lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=6,
                        capstyle="round")
    lc.set_array(vals[:-1])
    ax.add_collection(lc)

    # supports
    for sup in model.supports:
        x = model.nodes[sup.node].x
        y = max(seg.outer_d for seg in model.segments) * d_scale / 2
        if sup.kind == "fixed":
            ax.plot(x, y, marker="o", color="k", markersize=8)
        else:
            ax.plot(x, y, marker="^", color="k", markersize=8)

    # deformed shape overlay (deflection is scaled to be visible next to the span)
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
    ax.text(0.01, 0.95, f"diameters exaggerated x{d_scale:.0f} (not to scale)",
            transform=ax.transAxes, fontsize=8, color="0.3", verticalalignment="top")
    return lc


def plot_element(model, result, path, quantity="stress", title=None,
                 wind_speed=None, deformed=True):
    """Render the element colored by `quantity` and save to `path`.

    Geometry is not to scale: the tube diameters are exaggerated so the stepped
    telescoping profile is visible next to the element length. The deflected
    shape is overlaid with a labelled magnification factor. `wind_speed` (m/s)
    is shown in the title when given.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity: {quantity}")
    pts = _sample_quantity(model, result, quantity)
    vals = [p[1] * _DISPLAY[quantity] for p in pts]
    norm = Normalize(min(vals) if min(vals) != max(vals) else 0.0,
                     max(vals) if max(vals) != min(vals) else 1.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    lc = _draw_axis(ax, model, result, quantity, norm)
    fig.colorbar(lc, ax=ax, label=_LABELS[quantity], fraction=0.04, pad=0.02)
    _set_title(ax, title, wind_speed)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_es_pair(pin, sag, path, quantity="stress", title=None, wind_speed=None):
    """Render the rigid-pin (`pin`) and sag-rope (`sag`) ES models side by side
    on one figure with a shared colorbar.

    `pin` and `sag` are (model, result) tuples."""
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity: {quantity}")
    all_vals = []
    for model, result in (pin, sag):
        vals = [v * _DISPLAY[quantity] for _, v in _sample_quantity(model, result, quantity)]
        all_vals.extend(vals)
    norm = Normalize(min(all_vals) if min(all_vals) != max(all_vals) else 0.0,
                     max(all_vals) if max(all_vals) != min(all_vals) else 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             constrained_layout=True)
    lc = None
    for ax, (label, (model, result)) in zip(axes, (("rigid pin", pin), ("sag rope", sag))):
        lc = _draw_axis(ax, model, result, quantity, norm)
        ax.set_title(f"{label} support model", fontsize=9, loc="left")
    fig.colorbar(lc, ax=axes, label=_LABELS[quantity], fraction=0.04, pad=0.02)
    if title:
        fig.suptitle(_title_text(title, wind_speed), fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _set_title(ax, title, wind_speed):
    if title:
        ax.set_title(_title_text(title, wind_speed))


def _title_text(title, wind_speed):
    if wind_speed is not None:
        return f"{title}  (wind {wind_speed:.1f} m/s)"
    return title
