bipp
====


.. py:class:: Context(pu)

   Context that provides shared resources on a given processing unit.

   Args
      pu : str
          Either "CPU", "GPU" or "AUTO".

   .. rubric:: Attributes

   .. py:property:: processing_unit

      Returns
         The processing unit used by the context.

.. py:class:: Partition()

   Partition method.

   .. rubric:: Static Methods

   .. py:method:: auto()
      Automatic partition method through internal heuristic.

      Returns
         :py:class:`~bipp.Partition`

   .. py:method:: none()
      No partitioning.

      Returns
         :py:class:`~bipp.Partition`

   .. py:method:: grid(dimensions)
      Grid partitioning.

      Args
         dimensions : array-like(int)
             The grid dimensions of size three.

      Returns
         :py:class:`~bipp.Partition`


.. py:class:: NufftSynthesisOptions()

   Options for NUFFT Synthesis. Default settings are set initially.

   .. rubric:: Attributes

   .. py:property:: tolerance

      Returns
         Tolerance used when computing the NUFFT. Smaller value will increase accuracy but requires more operations.

   .. py:property:: collect_group_size

      Returns
         The maximum number of collected datasets processed together. Larger number typically improves performance but requires more memory. Internal heuristic is used if unset.

   .. py:property:: local_image_partition

      Returns
         The partition method used in the image domain. Partitioning decreases memory usage, but may come with a performance penalty.

   .. py:property:: local_uvw_partition

      Returns
         The partition method used in the UVW domain. Partitioning decreases memory usage, but may come with a performance penalty.

   .. rubric:: Methods

   .. py:method:: set_tolerance(tol)

      Args
         tol : float
             Tolerance used for NUFFT computation.
      Returns
         :py:class:`~bipp.NufftSynthesisOptions`

   .. py:method:: set_collect_group_size(size)

      Args
         size : int
             Collection group size. Must be at least 1.
      Returns
         :py:class:`~bipp.NufftSynthesisOptions`

   .. py:method:: set_local_image_partition(p)

      Args
         p : :py:class:`~bipp.Partition`
             Partition methid for image domain.
      Returns
         :py:class:`~bipp.NufftSynthesisOptions`

   .. py:method:: set_local_uvw_partition(p)

      Args
         p : :py:class:`~bipp.Partition`
             Partition methid for uvw domain.
      Returns
         :py:class:`~bipp.NufftSynthesisOptions`



.. py:class:: NufftSynthesis(ctx, n_antenna, n_beam, n_intervals, filter, lmn_x, lmn_y, lmn_z, precision, tol)

   Provides image generation using Bluebild with NUFFT Synthesis.

   Args
      ctx : :py:class:`~Context`
          Context to use resources from.
      n_antenna : int
          Number of antennas
      n_beam : int
          Number of beams.
      n_intervals : int
          Number of intverals to expect when collecting.
      filter : array-like(str)
          Array of filter to compute. Possible filter are "LSQ", "STD", "SQRT", "INV" and "INV_SQ".
      lmn_x : array-like(float)
          Array of image x coordinates.
      lmn_y : array-like(float)
          Array of image y coordinates.
      lmn_z : array-like(float)
          Array of image z coordinates.
      precision : str
          The precision to use in computations. Either "single" or "double".
      tol : float
          The tolerance used for nufft computation. A typical value is 0.001.

   .. rubric:: Methods

   .. py:method:: collect(n_eig, wl, intervals, w, xyz, uvw, s)

      Args
         n_eig : int
             Number of eigenvalues to compute.
         wl : float
             The wavelength.
         intervals : array_like(float)
             A 2D array of size (n_intervals, 2) for grouping energy data.
         w : array_like(complex)
             The beam forming matrix of size (n_antenna, n_beam).
         xyz : array_like(float)
             A 2D array of size (n_antenna, 3) of antenna postions.
         uvw : array_like(float)
             A 2D array of size (n_antenna^2, 3) of uvw coordinates.
         s : optional array_like(float)
             A 2D array of size (n_beam, n_beam) of visibilities. Optional.

   .. py:method:: get(f)

      Args
         f : str
             The filter to get images for.
      Returns
         :py:class:`~numpy.ndarray`
             (n_intervals, n_pixel) The image for each interval.


.. py:class:: StandardSynthesis(ctx, n_antenna, n_beam, n_intervals, filter, lmn_x, lmn_y, lmn_z, precision)

   Provides image generation using Bluebild with Standard Synthesis.

   Args
      ctx : :py:class:`~Context`
          Context to use resources from.
      n_antenna : int
          Number of antennas
      n_beam : int
          Number of beams.
      n_intervals : int
          Number of intverals to expect when collecting.
      filter : array-like(str)
          Array of filter to compute. Possible filter are "LSQ", "STD", "SQRT", "INV" and "INV_SQ".
      lmn_x : array-like(float)
          Array of image x coordinates.
      lmn_y : array-like(float)
          Array of image y coordinates.
      lmn_z : array-like(float)
          Array of image z coordinates.
      precision : str
          The precision to use in computations. Either "single" or "double".

   .. rubric:: Methods

   .. py:method:: collect(n_eig, wl, intervals, w, xyz, s)

      Args
         n_eig : int
             Number of eigenvalues to compute.
         wl : float
             The wavelength.
         intervals : array_like(float)
             A 2D array of size (n_intervals, 2) for grouping energy data.
         w : array_like(complex)
             The beam forming matrix of size (n_antenna, n_beam).
         xyz : array_like(float)
             A 2D array of size (n_antenna, 3) of antenna postions.
         s : optional array_like(float)
             A 2D array of size (n_beam, n_beam) of visibilities. Optional.

   .. py:method:: get(f)

      Args
         f : str
             The filter to get images for.
      Returns
         :py:class:`~numpy.ndarray`
             (n_intervals, n_pixel) The image for each interval.


Modules
^^^^^^^

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst
   :recursive:

   bipp.array
   bipp.beamforming
   bipp.frame
   bipp.gram
   bipp.instrument
   bipp.parameter_estimator
   bipp.source
   bipp.statistics
