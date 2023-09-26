r"""
Definition of the UVW frame and imaging grids.
"""

import numpy as np
import astropy.coordinates as aspy


def uvw_basis(field_center: aspy.SkyCoord) -> np.ndarray:
    r"""
    Transformation matrix associated to the local UVW frame.

    Args
        field_center: astropy.coordinates.SkyCoord
            Center of the FoV to which the local frame is attached.

    Returns
        uvw_frame: np.ndarray
            (3, 3) transformation matrix. Each column contains the ICRS coordinates of the U, V and W basis vectors defining the frame.
    """
    field_center_lon, field_center_lat = (
        field_center.data.lon.rad,
        field_center.data.lat.rad,
    )
    field_center_xyz = field_center.cartesian.xyz.value
    # UVW reference frame
    w_dir = field_center_xyz
    u_dir = np.array([-np.sin(field_center_lon), np.cos(field_center_lon), 0])
    v_dir = np.array(
        [
            -np.cos(field_center_lon) * np.sin(field_center_lat),
            -np.sin(field_center_lon) * np.sin(field_center_lat),
            np.cos(field_center_lat),
        ]
    )
    uvw_frame = np.stack((u_dir, v_dir, w_dir), axis=-1)
    return uvw_frame


def make_grids(grid_size, FoV, field_center):
    r"""
    Imaging grid.

    Args
        FoV: float
            Size of the FoV in radians.
        field_center: astropy.coordinates.SkyCoord
            Center of the FoV to which the local frame is attached.

    Returns
        lmn_grid, xyz_grid: Tuple[np.ndarray, np.ndarray]
            (3, grid_size * grid_size) grid coordinates in the local UVW frame and ICRS respectively.
            (3, grid_size, grid_size) image grid
    """
    lim = np.sin(FoV / 2)
    offset = lim / (grid_size / 2) * 0.5
    grid_slice1 = np.linspace(-lim - offset, lim - offset, grid_size)
    grid_slice2 = np.linspace(-lim + offset, lim + offset, grid_size)
    l_grid, m_grid = np.meshgrid(grid_slice2, grid_slice1)
    n_grid = np.sqrt(1 - l_grid**2 - m_grid**2)  # No -1 if r on the sphere !
    lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
    uvw_frame = uvw_basis(field_center)
    xyz_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
    lmn_grid = lmn_grid.reshape(3, -1)
    return lmn_grid, xyz_grid


def reshape_and_scale_uvw(wl, UVW):
    r"""
    Rescale by 2 * pi / wl and reshape to match NUFFT Synthesis expected input shape.

    Args
        wl: astropy.coordinates.SkyCoord
            Center of the FoV to which the local frame is attached.
        UVW: np.ndarray
            (N_antenna, N_antenna, 3) UVW coordinates expressed in the local UVW frame.

    Returns
        UVW: np.ndarray
            (N_antenna**2, 3) Rescaled and reshaped UVW as required by NUFFT Synthesis
    """
    # transpose because of coloumn major input format for bipp c++
    UVW = np.array(UVW.transpose((1, 0, 2)).reshape(-1, 3), order="F")
    UVW *= 2 * np.pi / wl
    return UVW
