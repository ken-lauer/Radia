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

Backends
--------
Two 3-D backends are available; matplotlib is the default (lightweight, headless-
friendly, always available).  PyVista / VTK gives correct occlusion and true
interactivity (rotate / zoom / pick), which is preferable for overlapping magnet
geometry -- inline in Jupyter, or a native window from a script::

    import uti_plot_radia as upr
    upr.view(obj, backend='pyvista', opacity=0.6)   # one-off, this call only
    upr.set_backend('pyvista')                       # route rad.ObjDrwOpenGL too

``plot_geometry_pyvista`` exposes ``color``, ``opacity``, ``show_edges`` etc.;
solid colours otherwise come from ``radia.ObjDrwAtr`` (both backends honour them).
PyVista is imported lazily, so this module still works where only matplotlib
(or neither) is installed.

In Jupyter, PyVista uses the interactive ``trame`` backend by default.  It is
smooth cell-by-cell but its async widgets clash with *Restart & Run All*; for a
run-all pass call ``upr.set_pyvista_backend('static')`` (plain image), then switch
back to ``'trame'``.  See the Notes in :func:`plot_geometry_pyvista`.
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
# PyVista / VTK rendering (interactive backend)
# ---------------------------------------------------------------------------
# matplotlib's 3-D axes use a painter's-algorithm depth sort with no true depth
# buffer, so overlapping magnets (undulator blocks, iron around coils) can bleed
# through.  VTK does real GPU occlusion and smooth interaction (rotate / zoom /
# pick), and RADIA's ObjDrwVTK output is already shaped to feed it.  This backend
# is optional -- imported lazily so matplotlib-only environments are unaffected.

def _section_to_polydata(section, pv, np, kind):
    """Build a ``pyvista.PolyData`` from one ObjDrwVTK section.

    Parameters
    ----------
    section : dict
        ``ObjDrwVTK(obj)['polygons']`` or ``['lines']`` (flat vertices / lengths /
        colors, as documented on :func:`_iter_polygons`).
    pv, np : module
        The ``pyvista`` and ``numpy`` modules (passed in to keep imports lazy).
    kind : {'faces', 'lines'}
        Whether the cells are filled polygons or polylines.

    Returns
    -------
    mesh : pyvista.PolyData
        Geometry with a per-cell ``'colors'`` uint8 RGB array (0..255) taken from
        the RADIA ``ObjDrwAtr`` colours.
    """
    verts = np.asarray(section['vertices'], dtype=float).reshape(-1, 3)

    # VTK connectivity: each cell is [n, i0, i1, ..., i(n-1)].  RADIA emits the
    # vertices of every polygon consecutively (not indexed), so the indices just
    # run in order; duplicate points at shared edges are harmless for display.
    conn = []
    i = 0
    for n in section['lengths']:
        conn.append(n)
        conn.extend(range(i, i + n))
        i += n
    conn = np.asarray(conn)

    mesh = pv.PolyData(verts, faces=conn) if kind == 'faces' \
        else pv.PolyData(verts, lines=conn)

    rgb = (np.asarray(section['colors'], dtype=float).reshape(-1, 3) * 255.0)
    mesh.cell_data['colors'] = rgb.astype(np.uint8)
    return mesh


def plot_geometry_pyvista(obj, color=None, opacity=1.0, show_edges=True,
                          edge_color='gray', line_width=3, background='white',
                          notebook=None, off_screen=False, screenshot=None,
                          window_size=(1024, 768), plotter=None, title=None,
                          show_axes=True, show=True, jupyter_backend=None):
    """Render RADIA object ``obj`` interactively with PyVista / VTK.

    Correct occlusion and interaction (rotate / zoom / pick) -- a better fit than
    matplotlib for overlapping magnet geometry.  In Jupyter it renders inline via
    trame; in a script it opens an interactive VTK window.

    Parameters
    ----------
    obj : int
        RADIA object index.
    color : color, optional
        Single colour for every solid (overrides RADIA's per-object ``ObjDrwAtr``
        colours).  When ``None`` (default) the RADIA colours are used.
    opacity : float, optional
        Solid opacity, 0 (transparent) .. 1 (opaque, default).  Lower it to see
        coils through iron.
    show_edges : bool, optional
        Draw polygon edges on the solids.
    edge_color : color, optional
        Colour of those edges.
    line_width : float, optional
        Width of the ``'lines'`` section (coil paths / object edges).
    background : color, optional
        Render-window background colour.
    notebook : bool, optional
        Force (``True``) or forbid (``False``) inline Jupyter rendering; ``None``
        lets PyVista auto-detect.
    off_screen : bool, optional
        Render without opening a window (needed for headless screenshots).
    screenshot : str, optional
        If given, save a PNG to this path.
    window_size : tuple of int, optional
        ``(width, height)`` in pixels for a new plotter.
    plotter : pyvista.Plotter, optional
        Draw into an existing plotter instead of creating one (e.g. to overlay
        several objects); when supplied, ``show`` is ignored.
    title : str, optional
        Corner text label.
    show_axes : bool, optional
        Show the orientation axes widget and a bounding-box grid with mm labels.
    show : bool, optional
        Display the plotter when this call created it.
    jupyter_backend : str, optional
        PyVista Jupyter backend for this display: ``'trame'`` (interactive,
        default) / ``'static'`` (plain image) / ``'client'`` / ``'html'``.
        ``None`` uses the module default (:func:`set_pyvista_backend`, ``'trame'``
        initially).  Ignored outside Jupyter.  See Notes.

    Returns
    -------
    plotter : pyvista.Plotter
        The plotter the geometry was drawn into.

    Notes
    -----
    **"Restart & Run All" and the trame backend.**  The default ``'trame'``
    backend (and ``'client'`` / ``'html'``) renders through trame's async Jupyter
    widgets.  Running cells one at a time is fine -- you get smooth interactive
    rotate / zoom / pick.  But when many geometry cells execute back-to-back under
    *Restart & Run All*, trame tries to set up a new render while the previous
    one's async context is still live on the kernel's event loop, raising::

        RuntimeError: cannot enter context: <Context> is already entered

    (plus ``Task was destroyed but it is pending`` / ``coroutine ... was never
    awaited``).  This is a trame + ipykernel event-loop race, pronounced on the
    very new Python 3.14; the geometry and this renderer are fine.  Options:

    * run the geometry cells individually (recommended -- keeps interactivity), or
    * switch to the image backend just for a run-all pass::

          uti_plot_radia.set_pyvista_backend('static')   # plain image, run-all safe

      then back to ``set_pyvista_backend('trame')`` for interactive exploration.
    """
    import radia as rad
    import numpy as np
    import pyvista as pv

    data = rad.ObjDrwVTK(obj)

    own = plotter is None
    if own:
        plotter = pv.Plotter(off_screen=off_screen, notebook=notebook,
                             window_size=window_size)
        plotter.set_background(background)

    polys = data.get('polygons')
    if polys and polys['lengths']:
        mesh = _section_to_polydata(polys, pv, np, 'faces')
        if color is not None:
            plotter.add_mesh(mesh, color=color, opacity=opacity,
                             show_edges=show_edges, edge_color=edge_color)
        else:
            plotter.add_mesh(mesh, scalars='colors', rgb=True, opacity=opacity,
                             show_edges=show_edges, edge_color=edge_color)

    lines = data.get('lines')
    if lines and lines['lengths']:
        lmesh = _section_to_polydata(lines, pv, np, 'lines')
        plotter.add_mesh(lmesh, scalars='colors', rgb=True, line_width=line_width)

    if title is not None:
        plotter.add_text(str(title), font_size=11)
    if show_axes:
        plotter.add_axes()
        try:
            plotter.show_grid(xtitle='X [mm]', ytitle='Y [mm]', ztitle='Z [mm]')
        except Exception:
            pass

    if screenshot is not None and not (own and show):
        plotter.screenshot(screenshot)
    if own and show:
        show_kw = {'screenshot': screenshot}
        # In Jupyter, pick a non-server backend by default so "Run All" (which
        # fires cells back-to-back) does not re-enter trame's async context.
        if _in_jupyter():
            jb = jupyter_backend or _PYVISTA_JUPYTER_BACKEND
            if jb is not None:
                show_kw['jupyter_backend'] = jb
        plotter.show(**show_kw)
    return plotter


def _in_jupyter():
    """Return ``True`` when running inside a Jupyter/ZMQ kernel (not a terminal).

    Used to decide whether to hand ``plotter.show`` a Jupyter backend.
    """
    try:
        ip = get_ipython()  # noqa: F821 -- injected by IPython
    except NameError:
        return False
    return ip is not None and type(ip).__name__ == 'ZMQInteractiveShell'


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND = 'matplotlib'   # what the patched ObjDrwOpenGL uses

# PyVista Jupyter backend used by plot_geometry_pyvista when it shows a plot in a
# notebook.  Default 'trame' = full interactivity (smooth rotate / zoom / pick),
# which is what you want for larger models.  The only run-all-safe option is
# 'static' (a plain inline image); every interactive backend ('trame'/'client'/
# 'html') goes through trame's async widgets -- see the Notes in
# plot_geometry_pyvista for the "Restart & Run All" caveat.
_PYVISTA_JUPYTER_BACKEND = 'trame'


def set_pyvista_backend(name):
    """Set the PyVista Jupyter backend used by :func:`plot_geometry_pyvista`.

    Parameters
    ----------
    name : {'trame', 'static', 'client', 'html', 'server', 'none', None}
        ``'trame'`` (default) gives full interactive rotate / zoom / pick.
        ``'static'`` is a plain inline image -- the only option safe under
        "Restart & Run All" (no trame async).  ``'client'`` / ``'html'`` are also
        interactive but, like ``'trame'``, go through trame's async widgets.
        ``None`` defers to PyVista's own global setting.  See the Notes in
        :func:`plot_geometry_pyvista`.

    Returns
    -------
    str or None
        The backend now in effect.
    """
    global _PYVISTA_JUPYTER_BACKEND
    _PYVISTA_JUPYTER_BACKEND = name
    return _PYVISTA_JUPYTER_BACKEND


def set_backend(name):
    """Choose which renderer the patched ``ObjDrwOpenGL`` (and :func:`view`) use.

    Parameters
    ----------
    name : {'matplotlib', 'mpl', 'pyvista', 'vtk'}
        ``matplotlib`` (default) is the lightweight, always-available, headless-
        friendly renderer; ``pyvista`` is the interactive VTK renderer.

    Returns
    -------
    str
        The normalised backend now in effect (``'matplotlib'`` or ``'pyvista'``).
    """
    global _DEFAULT_BACKEND
    key = str(name).lower()
    if key in ('matplotlib', 'mpl'):
        _DEFAULT_BACKEND = 'matplotlib'
    elif key in ('pyvista', 'vtk'):
        _DEFAULT_BACKEND = 'pyvista'
    else:
        raise ValueError("backend must be 'matplotlib' or 'pyvista', got %r" % name)
    return _DEFAULT_BACKEND


def get_backend():
    """Return the backend currently used by ``ObjDrwOpenGL`` / :func:`view`."""
    return _DEFAULT_BACKEND


def view(obj, backend=None, **kwargs):
    """Render ``obj`` with the selected (or default) 3-D backend.

    Parameters
    ----------
    obj : int
        RADIA object index.
    backend : {'matplotlib', 'pyvista', None}, optional
        Override the module default for this call only.
    **kwargs
        Passed through to :func:`plot_geometry` or :func:`plot_geometry_pyvista`.

    Returns
    -------
    object
        The matplotlib ``Axes3D`` or the ``pyvista.Plotter`` used.
    """
    key = str(backend or _DEFAULT_BACKEND).lower()
    if key in ('pyvista', 'vtk'):
        return plot_geometry_pyvista(obj, **kwargs)
    return plot_geometry(obj, **kwargs)


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
        # opens a separate native window; our replacement renders with whichever
        # backend set_backend() selected (matplotlib by default).
        if _DEFAULT_BACKEND == 'pyvista':
            return plot_geometry_pyvista(obj)
        # matplotlib path: in an IPython inline session display the figure
        # immediately so a later uti_plot ``pyplot.close('all')`` cannot discard
        # it before the cell renders (RADIA examples call ObjDrwOpenGL, then
        # uti_plot*).
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
