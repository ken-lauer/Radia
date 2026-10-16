"""Non-intrusive plotting fixes for RADIA on Linux / WSL and modern IPython.

This module makes RADIA's 3-D geometry viewing and the ``uti_plot`` field-plot
backend usable on platforms where the shipped code fails, **without touching
the compiled RADIA library or the existing ``uti_*`` source files**.  It fixes
two independent problems:

1. ``radia.ObjDrwOpenGL`` raises
   ``RuntimeError: This function is not implemented on that platform.``
   The interactive OpenGL viewer is only compiled into the Windows build, so on
   Linux / WSL there is no way to see the magnet geometry.  We replace the call
   at the Python level with a ``matplotlib`` renderer that draws the exact same
   polygons RADIA would send to OpenGL, obtained from ``radia.ObjDrwVTK``.

2. ``uti_plot_matplotlib.Backend._init_ipython`` calls
   ``get_ipython().magic('matplotlib ...')``.  The ``.magic()`` method was
   removed from IPython (>= 9, shipped alongside Python 3.14), so every
   ``uti_plot*`` call raises inside Jupyter / IPython.  We patch that one method
   to use the supported ``run_line_magic`` instead.

Nothing here modifies RADIA's physics functions -- only the presentation layer.

Parameters passing / return conventions follow numpydoc; see individual
functions.

Usage
-----
Run the examples from ``env/radia_python`` (already on ``sys.path``), or add the
directory explicitly, then import this module once after ``import radia``::

    import sys
    sys.path.append('/home/escott/Codes/radia/env/radia_python')
    import radia as rad
    import uti_plot_radia            # patches rad.ObjDrwOpenGL and uti_plot

    rad.ObjDrwOpenGL(obj)           # now renders via matplotlib instead of raising

The patched ``ObjDrwOpenGL`` accepts the same arguments as RADIA's and simply
builds a figure; the window is shown by the next ``uti_plot_show()`` (or by
``uti_plot_radia.show()``).  You can also bypass patching and call the renderer
directly::

    import uti_plot_radia
    uti_plot_radia.plot_geometry(obj, title='My magnet')
    uti_plot_radia.show()
"""

from __future__ import absolute_import, division, print_function

import sys


# ---------------------------------------------------------------------------
# 3-D geometry rendering (replacement for the Windows-only ObjDrwOpenGL)
# ---------------------------------------------------------------------------

def _is_ipython_inline():
    """Return ``True`` when running under IPython with an inline/notebook backend.

    Used to decide whether the patched ``ObjDrwOpenGL`` should push the geometry
    figure to the notebook immediately (so it survives a subsequent
    ``pyplot.close('all')`` inside ``uti_plot``).

    Returns
    -------
    bool
        ``True`` for the ``inline``/``nbAgg``/``ipympl`` backends inside IPython.
    """
    try:
        get_ipython()  # noqa: F821 -- injected into builtins by IPython
    except NameError:
        return False
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
    except Exception:
        return False
    return any(tag in backend for tag in ('inline', 'nbagg', 'ipympl', 'widget'))


def _iter_polygons(section):
    """Yield ``(vertices, color)`` for each polygon in a ``ObjDrwVTK`` section.

    RADIA's ``ObjDrwVTK`` returns, for both the ``'polygons'`` and ``'lines'``
    sub-dictionaries, three flat lists:

    * ``vertices`` -- ``[x0, y0, z0, x1, y1, z1, ...]`` (mm),
    * ``lengths``  -- number of vertices making up each polygon / polyline,
    * ``colors``   -- one ``[r, g, b]`` triple (0..1) per polygon / polyline.

    Parameters
    ----------
    section : dict
        One of ``ObjDrwVTK(obj)['polygons']`` or ``['lines']``.

    Yields
    ------
    verts : list of tuple
        ``[(x, y, z), ...]`` for a single polygon / polyline.
    color : tuple of float
        ``(r, g, b)`` in the 0..1 range for that polygon / polyline.
    """
    vertices = section['vertices']
    lengths = section['lengths']
    colors = section['colors']

    iv = 0   # running index into the flat vertex list
    ic = 0   # running index into the flat color list
    for n in lengths:
        verts = [
            (vertices[iv + 3 * k], vertices[iv + 3 * k + 1], vertices[iv + 3 * k + 2])
            for k in range(n)
        ]
        iv += 3 * n
        color = (colors[ic], colors[ic + 1], colors[ic + 2])
        ic += 3
        yield verts, color


def plot_geometry(obj, ax=None, draw_edges=True, edge_color='0.3',
                  line_color=None, axes=True, title=None, figsize=(9, 7),
                  show=False):
    """Render RADIA object ``obj`` in 3-D with matplotlib.

    This reproduces what ``radia.ObjDrwOpenGL`` shows on Windows, using the
    polygon soup returned by ``radia.ObjDrwVTK`` so the picture matches the
    colours set with ``radia.ObjDrwAtr``.

    Parameters
    ----------
    obj : int
        RADIA object index (as returned by ``ObjRecMag``, ``ObjCnt``, ...).
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Existing 3-D axes to draw into.  A new figure/axes is created if omitted.
    draw_edges : bool, optional
        Outline each solid polygon with ``edge_color`` for readability.
    edge_color : color, optional
        Matplotlib colour used for polygon outlines when ``draw_edges`` is set.
    line_color : color, optional
        Override colour for the ``'lines'`` section (coil paths, object edges).
        By default the colours RADIA supplies are used.
    axes : bool, optional
        Show the axis frame, ticks and labels (X/Y/Z in mm).
    title : str, optional
        Figure title.
    figsize : tuple of float, optional
        Figure size in inches when a new figure is created.
    show : bool, optional
        Call :func:`show` (``matplotlib.pyplot.show``) before returning.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes the geometry was drawn into.
    """
    # Imported lazily so that merely importing this module does not require
    # matplotlib (e.g. on a compute node that never plots).
    import radia as rad
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    data = rad.ObjDrwVTK(obj)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Track the bounding box so we can set a true (undistorted) aspect ratio.
    lo = [float('inf')] * 3
    hi = [float('-inf')] * 3

    def _grow(verts):
        for x, y, z in verts:
            for i, v in enumerate((x, y, z)):
                if v < lo[i]:
                    lo[i] = v
                if v > hi[i]:
                    hi[i] = v

    # Solid faces --------------------------------------------------------
    polys = data.get('polygons')
    if polys is not None and polys['lengths']:
        faces = []
        facecolors = []
        for verts, color in _iter_polygons(polys):
            faces.append(verts)
            facecolors.append(color)
            _grow(verts)
        coll = Poly3DCollection(faces, facecolors=facecolors)
        if draw_edges:
            coll.set_edgecolor(edge_color)
            coll.set_linewidth(0.3)
        ax.add_collection3d(coll)

    # Wire lines (coil trajectories, object edges) -----------------------
    # RADIA returns each edge loop as a single polyline with a variable number
    # of points.  Line3DCollection auto-scaling needs a homogeneous array, so we
    # split every polyline into consecutive 2-point segments.
    lines = data.get('lines')
    if lines is not None and lines['lengths']:
        segments = []
        seg_colors = []
        for verts, color in _iter_polygons(lines):
            col = line_color if line_color is not None else color
            for a, b in zip(verts[:-1], verts[1:]):
                segments.append((a, b))
                seg_colors.append(col)
            _grow(verts)
        if segments:
            ax.add_collection3d(Line3DCollection(segments, colors=seg_colors,
                                                 linewidths=0.6))

    # Framing ------------------------------------------------------------
    if lo[0] != float('inf'):
        spans = [hi[i] - lo[i] for i in range(3)]
        # Pad degenerate (flat) dimensions so the box aspect is well defined.
        spans = [s if s > 1e-9 else 1.0 for s in spans]
        for i in range(3):
            mid = 0.5 * (hi[i] + lo[i])
            half = 0.55 * spans[i]
            (ax.set_xlim, ax.set_ylim, ax.set_zlim)[i](mid - half, mid + half)
        try:
            ax.set_box_aspect(spans)   # matplotlib >= 3.3: true proportions
        except Exception:
            pass

    if axes:
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
    else:
        ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
    return ax


def show():
    """Show all pending matplotlib figures (thin wrapper over ``pyplot.show``)."""
    import matplotlib.pyplot as plt
    plt.show()


def savefig(fname, **kwargs):
    """Save the current matplotlib figure to ``fname`` (see ``pyplot.savefig``)."""
    import matplotlib.pyplot as plt
    plt.savefig(fname, **kwargs)


# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------

def patch_objdrwopengl():
    """Replace ``radia.ObjDrwOpenGL`` with the matplotlib renderer.

    The compiled ``ObjDrwOpenGL`` only works on Windows; on Linux / WSL it
    raises ``RuntimeError``.  After this call, ``radia.ObjDrwOpenGL(obj, ...)``
    builds a matplotlib figure instead (shown by the next ``uti_plot_show()``
    or :func:`show`).  Extra positional / keyword arguments accepted by RADIA's
    viewer are ignored.

    Returns
    -------
    bool
        ``True`` if the patch was applied.
    """
    import radia as rad

    def _obj_drw_opengl(obj, *args, **kwargs):
        # Signature-compatible stand-in for radia.ObjDrwOpenGL.  The real viewer
        # opens a separate native window; our replacement builds a matplotlib
        # figure.  In an IPython inline session we display it immediately so that
        # a later uti_plot ``pyplot.close('all')`` cannot discard it before the
        # cell renders (RADIA examples call ObjDrwOpenGL, then uti_plot*).
        ax = plot_geometry(obj, show=False)
        fig = ax.figure
        if _is_ipython_inline():
            try:
                from IPython.display import display
                import matplotlib.pyplot as plt
                display(fig)
                plt.close(fig)   # already shown inline; drop from the registry
            except Exception:
                pass
        return ax

    _obj_drw_opengl.__name__ = 'ObjDrwOpenGL'
    _obj_drw_opengl.__doc__ = 'matplotlib replacement for the Windows-only OpenGL viewer'
    rad.ObjDrwOpenGL = _obj_drw_opengl
    return True


def patch_uti_plot_ipython():
    """Fix the removed ``get_ipython().magic`` call in the uti_plot backend.

    ``uti_plot_matplotlib.Backend._init_ipython`` uses ``.magic()``, dropped in
    modern IPython.  This replaces that single method with one that prefers the
    supported ``run_line_magic`` and falls back to ``.magic`` on old IPython.
    The compiled RADIA code and the rest of ``uti_plot*`` are untouched.

    Returns
    -------
    bool
        ``True`` if the patch was applied (or already present); ``False`` if the
        matplotlib backend module could not be imported.
    """
    try:
        import uti_plot_matplotlib as upm
    except Exception:
        return False

    backend_cls = upm.Backend
    if getattr(backend_cls, '_radia_magic_patched', False):
        return True

    try:
        import uti_plot as up
        default_backend = up.DEFAULT_BACKEND
    except Exception:
        default_backend = '<default>'

    def _init_ipython(self, backend):
        # Mirror of the original method, but using run_line_magic.  ``get_ipython``
        # is injected into builtins by IPython, so it resolves here as it did in
        # the original module.
        if backend == default_backend:
            backend = 'inline'
        is_mpld3 = backend == 'mpld3'
        b = 'inline' if is_mpld3 else backend
        ip = get_ipython()  # noqa: F821 -- provided by IPython at runtime
        if hasattr(ip, 'run_line_magic'):
            ip.run_line_magic('matplotlib', b)
        else:                       # very old IPython
            ip.magic('matplotlib ' + b)
        if is_mpld3:
            try:
                import mpld3
                mpld3.enable_notebook()
            except Exception:
                print('mpld3: backend unavailable; plots will be inline')
                backend = 'inline'
        return backend

    backend_cls._init_ipython = _init_ipython
    backend_cls._radia_magic_patched = True
    return True


def apply_all(verbose=False):
    """Apply every patch this module provides.

    Parameters
    ----------
    verbose : bool, optional
        Print a one-line summary of what was patched.

    Returns
    -------
    dict
        Mapping of patch name to the boolean result.
    """
    results = {
        'ObjDrwOpenGL': patch_objdrwopengl(),
        'uti_plot_ipython': patch_uti_plot_ipython(),
    }
    if verbose:
        print('uti_plot_radia: ' + ', '.join(
            '%s=%s' % (k, 'ok' if v else 'skipped') for k, v in results.items()))
    return results


# Apply automatically on import so a single ``import uti_plot_radia`` is enough.
_PATCH_RESULTS = apply_all(verbose=False)
