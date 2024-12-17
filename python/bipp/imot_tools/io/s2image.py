# #############################################################################
# s2image.py
# ==========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

r"""
Visualization and storage of images on :math:`\mathbb{S}^{2}`.
"""

import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
import matplotlib.axes as axes
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pyproj
import scipy.linalg as linalg

import bipp.imot_tools.io.fits as ifits
import bipp.imot_tools.io.plot as plot
import bipp.imot_tools.math.sphere.transform as transform
import bipp.imot_tools.util.argcheck as chk


def from_fits(file_name):
    """
    Load image from FITS file.

    Images must have been saved by calling :py:meth:`~imot_tools.io.s2image.Image.to_fits`.

    Parameters
    ----------
    file_name : path-like
        Name of file.

    Returns
    -------
    I : :py:class:`~imot_tools.io.s2image.Image`
    """
    with fits.open(
        file_name, mode="readonly", memmap=True, lazy_load_hdus=True
    ) as hdulist:
        # PrimaryHDU: grid / class info
        primary_hdu = hdulist[0]
        image_hdu = hdulist["IMAGE"]
        klass = globals()[primary_hdu.header["IMG_TYPE"]]

        I = klass._from_fits(primary_hdu, image_hdu)
        return I


class Image:
    r"""
    Container for storing real-valued images defined on :math:`\mathbb{S}^{2}`.

    Main features:

    * import/export spherical maps to (non-standard) FITS format;
    * advanced 2D plotting based on `Matplotlib <https://matplotlib.org/>`_;

    Examples
    --------
    .. doctest::

       import numpy as np

       import bipp.imot_tools.io.s2image as s2image
       import bipp.imot_tools.math.func as func
       import bipp.imot_tools.math.sphere.grid as grid
       import bipp.imot_tools.math.sphere.transform as transform

       # grid settings =======================
       direction = transform.eq2cart(1, lat=np.deg2rad(30), lon=np.deg2rad(20)).reshape(-1)
       FoV = np.deg2rad(60)
       N_height, N_width = 256, 384
       px_grid = grid.uniform(direction, FoV, size=[N_height, N_width])

       # data settings =======================
       beta0, a0 = 0.7, [1, 1, 1]
       beta1, a1 = 0.9, [0, 0, 1]
       kent0 = func.Kent(k=func.Kent.min_scale(FoV, beta0) * 2,
                         beta=beta0,
                         g1=direction,
                         a=a0)
       kent1 = func.Kent(k=func.Kent.min_scale(FoV, beta1) * 2,
                         beta=beta1,
                         g1=direction,
                         a=a1)

       data0 = (kent0(px_grid.reshape(3, N_height * N_width).T)
                .reshape(N_height, N_width))
       data1 = (kent1(px_grid.reshape(3, N_height * N_width).T)
                .reshape(N_height, N_width))
       data = np.stack([data0, data1], axis=0)

       # Image creation ======================
       I = s2image.Image(data, px_grid)

    Data IO:

    .. doctest::

       I.to_fits('test.fits')  # save to FITS
       I2 = s2image.from_fits('test.fits')  # load from FITS

    Interactive plotting:

    .. doctest::

       kwargs = dict(cmap='jet')
       I.draw(data_kwargs=kwargs)  # AEQD projection by default, all layers.

    .. image:: _img/sphericalimage_aeqd_example.png

    .. doctest::

       kwargs = dict(cmap='jet')
       I.draw(index=0, projection='GNOM', data_kwargs=kwargs)  # Only show first data slice.

    .. image:: _img/sphericalimage_gnom_example.png

    .. doctest::

       kwargs = dict(cmap='jet')
       I.draw(index=1, projection='LCC', data_kwargs=kwargs)

    .. image:: _img/sphericalimage_lcc_example.png
    """

    @chk.check(dict(data=chk.has_reals, grid=chk.has_reals))
    def __init__(self, data, grid):
        """
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            multi-level (float) data-cube.

            Possible shapes are:

            * (N_height, N_width);
            * (N_image, N_height, N_width);
            * (N_points,);
            * (N_image, N_points).
        grid : :py:class:`~numpy.ndarray`
            (3, ...) Cartesian coordinates of the sky on which the data points are defined.

            Possible shapes are:

            * (3, N_height, N_width);
            * (3, N_points).

        Notes
        -----
        For efficiency reasons, `data` and `grid` are not copied internally.
        """
        grid = np.asarray(grid)
        grid_shape_error_msg = (
            "Parameter[grid] must have shape (3, N_height, N_width) or (3, N_points)."
        )
        if len(grid) != 3:
            raise ValueError(grid_shape_error_msg)
        if grid.ndim == 2:
            self._is_gridded = False
        elif grid.ndim == 3:
            self._is_gridded = True
        else:
            raise ValueError(grid_shape_error_msg)
        self._grid = grid / linalg.norm(grid, axis=0)

        data = np.asarray(data)
        if self._is_gridded:
            N_height, N_width = self._grid.shape[1:]
            if (data.ndim == 2) and chk.has_shape([N_height, N_width])(data):
                self._data = data[np.newaxis]
            elif (data.ndim == 3) and chk.has_shape([N_height, N_width])(data[0]):
                self._data = data
            else:
                raise ValueError("Parameters[grid, data] are inconsistent.")
        else:
            N_points = self._grid.shape[1]
            if (data.ndim == 1) and chk.has_shape([N_points])(data):
                self._data = data[np.newaxis]
            elif (data.ndim == 2) and chk.has_shape([N_points])(data[0]):
                self._data = data
            else:
                raise ValueError("Parameters[grid, data] are inconsistent.")

    @property
    def data(self):
        """
        Returns
        -------
        I : :py:class:`~numpy.ndarray`
            (N_image, ...) data cube.
        """
        return self._data

    @property
    def grid(self):
        """
        Returns
        -------
        XYZ : :py:class:`~numpy.ndarray`
            (3, ...) Cartesian coordinates of the grid on which the data points are defined.
        """
        return self._grid

    def to_fits(self, file_name):
        """
        Save image to FITS file.

        Parameters
        ----------
        file_name : path-like
            Name of file.

        Notes
        -----
        * :py:class:`~imot_tools.io.s2image.Image` subclasses may write WCS information to the FITS
          file.  The user-provided `grid` is assumed in ICRS.  If this is not the case, rotate the
          grid accordingly before calling :py:meth:`~imot_tools.io.s2image.Image.to_fits`.

        * Data cubes are stored in a secondary IMAGE frame and can be viewed with DS9 using::

              $ ds9 <FITS_file>.fits[IMAGE]

          Only FITS-compliant maps are successfully visualized with DS9.
          Moreover WCS information only available in select subclasses.
        """
        primary_hdu = self._PrimaryHDU()
        image_hdu = self._ImageHDU()

        hdulist = fits.HDUList([primary_hdu, image_hdu])
        hdulist.writeto(file_name, overwrite=True)

    def _PrimaryHDU(self):
        """
        Generate primary Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        """
        metadata = dict(IMG_TYPE=(self.__class__.__name__, "Image subclass"))

        # grid: stored as angles to reduce file size.
        _, colat, lon = transform.cart2pol(*self._grid)
        coordinates = np.stack([colat, lon], axis=0)

        hdu = fits.PrimaryHDU(data=coordinates)
        for k, v in metadata.items():
            hdu.header[k] = v
        return hdu

    def _ImageHDU(self):
        """
        Generate image Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.ImageHDU`
        """
        hdu = fits.ImageHDU(data=self._data, name="IMAGE")
        return hdu

    @classmethod
    @chk.check(
        dict(
            primary_hdu=chk.is_instance(fits.PrimaryHDU),
            image_hdu=chk.is_instance(fits.ImageHDU),
        )
    )
    def _from_fits(cls, primary_hdu, image_hdu):
        """
        Load image from Header Descriptor Units.

        Parameters
        ----------
        primary_hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        image_hdu : :py:class:`~astropy.io.fits.ImageHDU`

        Returns
        -------
        I : :py:class:`~imot_tools.io.s2image.Image`
        """
        # PrimaryHDU: grid specification.
        colat, lon = primary_hdu.data
        grid = transform.pol2cart(1, colat, lon)

        # ImageHDU: extract data cube.
        data = image_hdu.data

        I = cls(data=data, grid=grid)
        return I

    @property
    def shape(self):
        """
        Returns
        -------
        sh : tuple
            Shape of data cube.
        """
        return self._data.shape

    @chk.check(
        dict(
            index=chk.accept_any(
                chk.is_integer, chk.has_integers, chk.is_instance(slice)
            ),
            projection=chk.is_instance(str),
            catalog=chk.allow_None(chk.has_reals),
            show_gridlines=chk.is_boolean,
            show_colorbar=chk.is_boolean,
            ax=chk.allow_None(chk.is_instance(axes.Axes)),
            use_contours=chk.is_boolean,
            data_kwargs=chk.allow_None(chk.is_instance(dict)),
            grid_kwargs=chk.allow_None(chk.is_instance(dict)),
            catalog_kwargs=chk.allow_None(chk.is_instance(dict)),
        )
    )
    def draw(
        self,
        index=slice(None),
        projection="AEQD",
        catalog=None,
        show_gridlines=True,
        show_colorbar=True,
        ax=None,
        use_contours=False,
        data_kwargs=None,
        grid_kwargs=None,
        catalog_kwargs=None,
    ):
        """
        Plot spherical image using a 2D projection.

        Parameters
        ----------
        index : int, array-like(int), slice
            Slices of the data-cube to show.

            If multiple layers are provided, they are summed together.
        projection : str
            Plot projection.

            Must be one of (case-insensitive):

            * AEQD: `Azimuthal Equi-Distant <https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection>`_; (default)
            * LAEA: `Lambert Equal-Area <https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection>`_;
            * LCC: `Lambert Conformal Conic <https://en.wikipedia.org/wiki/Lambert_conformal_conic_projection>`_;
            * ROBIN: `Robinson <https://en.wikipedia.org/wiki/Robinson_projection>`_;
            * GNOM: `Gnomonic <https://en.wikipedia.org/wiki/Gnomonic_projection>`_;
            * HEALPIX: `Hierarchical Equal-Area Pixelisation <https://en.wikipedia.org/wiki/HEALPix>`_.

            Notes
            -----
            * (AEQD, LAEA, LCC, GNOM) are recommended for mapping portions of the sphere.

                * LCC breaks down when mapping polar regions.

            * (ROBIN, HEALPIX) are recommended for mapping the entire sphere.
        catalog : :py:class:`~numpy.ndarray`
            (3, N_src) source directions to overlay on top of images. (Default: no overlay)
            The catalog is assumed to lie in the same reference frame as `grid`.
        show_gridlines : bool
            Show RA/DEC gridlines. (Default: True)
        show_colorbar : bool
            Show colorbar. (Default: True)
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.

            If :py:obj:`None`, a new axes is used.
        use_contours: bool

            * If :py:obj:`True`, use [tri]contourf() to produce the plots.
            * If :py:obj:`False` (default), use [tri]pcolor[mesh]() to produce the plots.
        data_kwargs : dict
            Keyword arguments related to data-cube visualization.

            Depending on `use_contours`, accepted keys are:

            * :py:meth:`~matplotlib.axes.Axes.contourf` / :py:meth:`~matplotlib.axes.Axes.pcolormesh` options;
            * :py:meth:`~matplotlib.axes.Axes.tricontourf` / :py:meth:`~matplotlib.axes.Axes.tripcolor` options.
        grid_kwargs : dict
            Keyword arguments related to grid visualization.

            Accepted keys are:

            * N_parallel : int
                Number declination lines to show in viewable region. (Default: 3)
            * N_meridian : int
                Number of right-ascension lines to show in viewable region. (Default: 3)
            * polar_plot : bool
                Correct RA/DEC gridlines when mapping polar regions. (Default: False)

                When mapping polar regions, meridian lines may be doubled at 180W/E, making it seem like a meridian line is missing.
                Setting `polar_plot` to :py:obj:`True` redistributes the meridians differently to correct the issue.

                This option only makes sense when mapping polar regions, and will produce incorrect gridlines otherwise.
            * ticks : bool
                Add RA/DEC labels next to gridlines. (Default: False)
                TODO: change to True once implemented
        catalog_kwargs : dict
            Keyword arguments related to catalog visualization.

            Accepted keys are:

            * :py:meth:`~matplotlib.axes.Axes.scatter` options.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`
        proj : :py:class:`pyproj.Proj`
        scm : :py:class:`~matplotlib.cm.ScalarMappable`
        """
        if ax is None:
            fig, ax = plt.subplots()

        proj = self._draw_projection(projection)
        scm = self._draw_data(index, data_kwargs, use_contours, proj, ax)
        cbar = self._draw_colorbar(show_colorbar, scm, ax)
        self._draw_gridlines(show_gridlines, grid_kwargs, proj, ax)
        self._draw_catalog(catalog, catalog_kwargs, proj, ax)
        self._draw_beautify(proj, ax)

        return ax, proj, scm

    def _draw_projection(self, projection):
        """
        Setup :py:class:`pyproj.Proj` object to do (lon,lat) <-> (x,y) transforms.

        Parameters
        ----------
        projection : str
            `projection` parameter given to :py:meth:`draw`.

        Returns
        -------
        proj : :py:class:`pyproj.Proj`
        """
        # Most projections can be provided a point in space around which distortions are minimized.
        # We choose this point to approximately map to the center of the grid when appropriate.
        # (approximate since it is not always a spherical cap.)
        if self._is_gridded:  # (3, N_height, N_width) grid
            grid_dir = np.mean(self._grid, axis=(1, 2))
        else:  # (3, N_points) grid
            grid_dir = np.mean(self._grid, axis=1)
        _, grid_lat, grid_lon = transform.cart2eq(*grid_dir)
        grid_lat, grid_lon = self._wrapped_rad2deg(grid_lat, grid_lon)

        p_name = projection.lower()
        if p_name == "lcc":
            # Lambert Conformal Conic
            proj = pyproj.Proj(proj="lcc", lon_0=grid_lon, lat_0=grid_lat, R=1)
        elif p_name == "aeqd":
            # Azimuthal Equi-Distant
            proj = pyproj.Proj(proj="aeqd", lon_0=grid_lon, lat_0=grid_lat, R=1)
        elif p_name == "laea":
            # Lambert Equal-Area
            proj = pyproj.Proj(proj="laea", lon_0=grid_lon, lat_0=grid_lat, R=1)
        elif p_name == "robin":
            # Robinson
            proj = pyproj.Proj(proj="robin", lon_0=grid_lon, R=1)
        elif p_name == "gnom":
            # Gnomonic
            proj = pyproj.Proj(proj="gnom", lon_0=grid_lon, lat_0=grid_lat, R=1)
        elif p_name == "healpix":
            # Hierarchical Equal-Area Pixelisation
            proj = pyproj.Proj(proj="healpix", lon_0=grid_lon, lat_0=grid_lat, R=1)
        else:
            raise ValueError(
                "Parameter[projection] is not a valid projection specifier."
            )

        return proj

    @chk.check(
        dict(
            index=chk.accept_any(
                chk.is_integer, chk.has_integers, chk.is_instance(slice)
            ),
            data_kwargs=chk.allow_None(chk.is_instance(dict)),
            use_contours=chk.is_boolean,
            projection=chk.is_instance(pyproj.Proj),
            ax=chk.is_instance(axes.Axes),
        )
    )
    def _draw_data(self, index, data_kwargs, use_contours, projection, ax):
        """
        Contour plot of data.

        Parameters
        ----------
        index : int, array-like(int), slice
            `index` parameter given to :py:meth:`draw`.
        data_kwargs : dict
            `data_kwargs` parameter given to :py:meth:`draw`.
        use_contours: bool
            If :py:obj:`True`, use [tri]contourf() to produce the plots.
            If :py:obj:`False`, use [tri]pcolor[mesh]() to produce the plots.
        projection : :py:class:`~pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.

        Returns
        -------
        scm : :py:class:`~matplotlib.cm.ScalarMappable`
        """
        if data_kwargs is None:
            data_kwargs = dict()

        N_image = self.shape[0]
        if chk.is_integer(index):
            index = np.array([index], dtype=int)
        elif chk.has_integers(index):
            index = np.array(index, dtype=int)
        else:  # slice()
            index = np.arange(N_image, dtype=int)[index]
            if index.size == 0:
                raise ValueError("No data-cube slice chosen.")
        if not np.all((0 <= index) & (index < N_image)):
            raise ValueError("Parameter[index] is out of bounds.")
        data = np.sum(self._data[index], axis=0)

        _, grid_lat, grid_lon = transform.cart2eq(*self._grid)
        grid_x, grid_y = self._eq2xy(grid_lat, grid_lon, projection)

        # Colormap choice
        if "cmap" in data_kwargs:
            obj = data_kwargs.pop("cmap")
            if chk.is_instance(str)(obj):
                cmap = cm.get_cmap(obj)
            else:
                cmap = obj
        else:
            cmap = cm.get_cmap("RdPu")

        if self._is_gridded:
            if use_contours:
                scm = ax.contourf(
                    grid_x, grid_y, data, cmap.N, cmap=cmap, **data_kwargs
                )
            else:
                scm = ax.pcolormesh(grid_x, grid_y, data, cmap=cmap, **data_kwargs)
        else:
            triangulation = tri.Triangulation(grid_x, grid_y)
            if use_contours:
                scm = ax.tricontourf(
                    triangulation, data, cmap.N, cmap=cmap, **data_kwargs
                )
            else:
                scm = ax.tripcolor(triangulation, data, cmap=cmap, **data_kwargs)

        # Show coordinates in status bar
        def sexagesimal_coords(x, y):
            lon, lat = projection(x, y, errcheck=False, inverse=True)
            lon = (
                coord.Angle(lon * u.deg)
                .wrap_at(180 * u.deg)
                .to_string(unit=u.hourangle, sep="hms")
            )
            lat = coord.Angle(lat * u.deg).to_string(unit=u.degree, sep="dms")

            msg = f"RA: {lon}, DEC: {lat}"
            return msg

        ax.format_coord = sexagesimal_coords

        return scm

    @chk.check(
        dict(
            show_colorbar=chk.is_boolean,
            scm=chk.is_instance(cm.ScalarMappable),
            ax=chk.is_instance(axes.Axes),
        )
    )
    def _draw_colorbar(self, show_colorbar, scm, ax):
        """
        Attach colorbar.

        Parameters
        ----------
        show_colorbar : bool
            `show_colorbar` parameter given to :py:meth:`draw`.
        scm : :py:class:`~matplotlib.cm.ScalarMappable`
            Intensity scale.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.

        Returns
        -------
        cbar : :py:class:`~matplotlib.colorbar.Colorbar`
        """
        if show_colorbar:
            cbar = plot.colorbar(scm, ax)
        else:
            cbar = None

        return cbar

    @chk.check(
        dict(
            show_gridlines=chk.is_boolean,
            grid_kwargs=chk.allow_None(chk.is_instance(dict)),
            projection=chk.is_instance(pyproj.Proj),
            ax=chk.is_instance(axes.Axes),
        )
    )
    def _draw_gridlines(self, show_gridlines, grid_kwargs, projection, ax):
        """
        Plot Right-Ascension / Declination lines.

        Parameters
        ----------
        show_gridlines : bool
            `show_gridlines` parameter given to :py:meth:`draw`.
        grid_kwargs : dict
            `grid_kwargs` parameter given to :py:meth:`draw`.
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.
        """
        if grid_kwargs is None:
            grid_kwargs = dict()

        if "N_parallel" in grid_kwargs:
            N_parallel = grid_kwargs.pop("N_parallel")
            if not (chk.is_integer(N_parallel) and (N_parallel >= 3)):
                raise ValueError("Value[N_parallel] must be at least 3.")
        else:
            N_parallel = 3

        if "N_meridian" in grid_kwargs:
            N_meridian = grid_kwargs.pop("N_meridian")
            if not (chk.is_integer(N_meridian) and (N_meridian >= 3)):
                raise ValueError("Value[N_meridian] must be at least 3.")
        else:
            N_meridian = 3

        if "polar_plot" in grid_kwargs:
            polar_plot = grid_kwargs.pop("polar_plot")
            if not chk.is_boolean(polar_plot):
                raise ValueError("Value[polar_plot] must be boolean.")
        else:
            polar_plot = False

        if "ticks" in grid_kwargs:
            show_ticks = grid_kwargs.pop("ticks")
            if not chk.is_boolean(show_ticks):
                raise ValueError("Value[ticks] must be boolean.")
        else:
            # TODO: change to True once implemented.
            show_ticks = False

        plot_style = dict(alpha=0.5, color="k", linewidth=1, linestyle="solid")
        plot_style.update(grid_kwargs)

        _, grid_lat, grid_lon = transform.cart2eq(*self._grid)
        grid_lat, grid_lon = self._wrapped_rad2deg(grid_lat, grid_lon)

        # RA curves
        meridian = dict()
        dec_span = np.linspace(grid_lat.min(), grid_lat.max(), 200)
        if polar_plot:
            ra = np.linspace(-180, 180, N_meridian, endpoint=False)
        else:
            ra = np.linspace(grid_lon.min(), grid_lon.max(), N_meridian)
        for _ in ra:
            ra_span = _ * np.ones_like(dec_span)
            grid_x, grid_y = self._eq2xy(
                np.deg2rad(dec_span), np.deg2rad(ra_span), projection
            )

            if show_gridlines:
                mer = ax.plot(grid_x, grid_y, **plot_style)[0]
                meridian[_] = mer

        # DEC curves
        parallel = dict()
        ra_span = np.linspace(grid_lon.min(), grid_lon.max(), 200)
        if polar_plot:
            dec = np.linspace(grid_lat.min(), grid_lat.max(), N_parallel + 1)
        else:
            dec = np.linspace(grid_lat.min(), grid_lat.max(), N_parallel)
        for _ in dec:
            dec_span = _ * np.ones_like(ra_span)
            grid_x, grid_y = self._eq2xy(
                np.deg2rad(dec_span), np.deg2rad(ra_span), projection
            )

            if show_gridlines:
                par = ax.plot(grid_x, grid_y, **plot_style)[0]
                parallel[_] = par

        # LAT/LON ticks
        if show_gridlines and show_ticks:
            raise NotImplementedError("Not yet implemented.")

    @chk.check(
        dict(
            catalog=chk.allow_None(chk.has_reals),
            projection=chk.is_instance(pyproj.Proj),
            ax=chk.is_instance(axes.Axes),
        )
    )
    def _draw_catalog(self, catalog, catalog_kwargs, projection, ax):
        """
        Overlay catalog on top of map.

        Parameters
        ----------
        catalog : :py:class:`~numpy.ndarray`
            `catalog` parameter given to :py:meth:`draw`.
        catalog_kwargs : dict
            `catalog_kwargs` parameter given to :py:meth:`draw`.
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to plot on.
        """
        if catalog is not None:
            N_src = catalog.size // 3
            if not (catalog.shape == (3, N_src)):
                raise ValueError("Parameter[catalog]: expected (3, N_src) array.")

            _, c_lat, c_lon = transform.cart2eq(*catalog)
            c_x, c_y = self._eq2xy(c_lat, c_lon, projection)

            if catalog_kwargs is None:
                catalog_kwargs = dict()

            plot_style = dict(s=400, facecolors="none", edgecolors="w")
            plot_style.update(catalog_kwargs)

            ax.scatter(c_x, c_y, **plot_style)

    @chk.check(
        dict(projection=chk.is_instance(pyproj.Proj), ax=chk.is_instance(axes.Axes))
    )
    def _draw_beautify(self, projection, ax):
        """
        Format plot.

        Parameters
        ----------
        projection : :py:class:`pyproj.Proj`
            PyProj projection object.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.
        """
        ax.axis("off")
        ax.axis("equal")

    @classmethod
    def _wrapped_rad2deg(cls, lat_r, lon_r):
        """
        Equatorial coordinate [rad] -> [deg] unit conversion.
        Output longitude guaranteed to lie in [-180, 180) [deg].

        Parameters
        ----------
        lat_r : :py:class:`~numpy.ndarray`
        lon_r : :py:class:`~numpy.ndarray`

        Returns
        -------
        lat_d : :py:class:`~numpy.ndarray`
        lon_d : :py:class:`~numpy.ndarray`
        """
        lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
        lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
        return lat_d, lon_d

    @classmethod
    def _eq2xy(cls, lat_r, lon_r, projection):
        """
        Transform (lon,lat) [rad] to (x,y).
        Some projections have unmappable regions or exhibit singularities at certain points.
        These regions are colored white in contour plots by replacing their incorrect value (1e30)
        with NaN.

        Parameters
        ----------
        lat_r : :py:class:`~numpy.ndarray`
        lon_r : :py:class:`~numpy.ndarray`
        projection : :py:class:`~pyproj.Proj`

        Returns
        -------
        x : :py:class:`~numpy.ndarray`
        y : :py:class:`~numpy.ndarray`
        """
        lat_d, lon_d = cls._wrapped_rad2deg(lat_r, lon_r)
        x, y = projection(lon_d, lat_d, errcheck=False)
        x[np.isclose(x, 1e30)] = np.nan
        y[np.isclose(y, 1e30)] = np.nan
        return x, y


class WCSImage(Image):
    """
    Spherical image container for WCS-described maps.

    Main features:

    * import/export to FITS-compliant format;
    * View images using DS9/AstroPy. (see :py:meth:`~imot_tools.io.s2image.Image.to_fits`)
    """

    @chk.check("WCS", chk.is_instance(wcs.WCS))
    def __init__(self, data, WCS):
        """
        Parameters
        ----------
        data : :py:class:`~numpy.ndarray`
            multi-level (float) data-cube.

            Possible shapes are:

            * (N_height, N_width);
            * (N_image, N_height, N_width);

        WCS : :py:class:`~astropy.wcs.WCS`
            (N_width, N_height) World Coordinate System with spatial axes.

        Notes
        -----
        Caution: the FITS standard uses FORTRAN conventions for index/array
        ordering. `WCS` dimensions/information (FORTRAN-ordered) must therefore
        be reversed compared to `data` (assumed C-ordered).
        """
        if WCS.naxis != 2:
            raise ValueError("Parameter[WCS] is ambiguous.")

        N_width, N_height = WCS.array_shape
        if data.shape[-2:] != (N_height, N_width):
            raise ValueError("Parameters[grid, data] are inconsistent.")

        grid = ifits.pix_grid(WCS)  # (3, N_height, N_width)
        super().__init__(data, grid)

        self._WCS = WCS.sub([1, 2, 0])  # 0 = insert new axis.
        self._WCS.wcs.cname[2] = "LAYER"
        self._WCS.wcs.ctype[2] = ""

    @property
    def WCS(self):
        """
        Returns
        -------
        WCS : :py:class:`~astropy.wcs.WCS`
            World Coordinate System (WCS), augmented with extra axes for
            multi-layer images.
        """
        return self._WCS

    def _PrimaryHDU(self):
        """
        Generate primary Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        """
        metadata = dict(IMG_TYPE=(self.__class__.__name__, "Image subclass"))

        hdu = fits.PrimaryHDU()
        for k, v in metadata.items():
            hdu.header[k] = v
        return hdu

    def _ImageHDU(self):
        """
        Generate image Header Descriptor Unit (HDU) for FITS export.

        Returns
        -------
        hdu : :py:class:`~astropy.io.fits.ImageHDU`
        """
        hdu = fits.ImageHDU(data=self._data, header=self._WCS.to_header(), name="IMAGE")
        return hdu

    @classmethod
    @chk.check(
        dict(
            primary_hdu=chk.is_instance(fits.PrimaryHDU),
            image_hdu=chk.is_instance(fits.ImageHDU),
        )
    )
    def _from_fits(cls, primary_hdu, image_hdu):
        """
        Load image from Header Descriptor Units.

        Parameters
        ----------
        primary_hdu : :py:class:`~astropy.io.fits.PrimaryHDU`
        image_hdu : :py:class:`~astropy.io.fits.ImageHDU`

        Returns
        -------
        I : :py:class:`~imot_tools.io.s2image.Image`
        """
        # PrimaryHDU: nothing for WCSImage.

        # ImageHDU: extract data cube.
        data = image_hdu.data
        WCS = wcs.WCS(image_hdu).celestial  # Must drop LAYER-dimension to
        # avoid ambiguity in constructor.

        I = cls(data=data, WCS=WCS)
        return I
