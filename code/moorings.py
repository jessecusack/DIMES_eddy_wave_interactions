import numpy as np
import gsw
import scipy.interpolate as ipl


def interp_quantity(z, q, zi, pref, return_diagnostics=False):
    """To correct for mooring motion and aid calculations, mooring data is
    often interpolated to specific depth or pressure. This function achieves
    that using the method of Philips and modified by Brearly.

    It is assumed that the data consists of time series from instruments at
    various depths represented by 2D numpy arrays, A[i, j], where the index i
    determins the depth and j the time. For example,

    A = np.array([[1., 1.2, 1.3],
                  [0.5, 0.7, 0.8]])

    would describe a mooring with 2 instruments levels and each containing 3
    measurements.

    Input
    -----
    z : 2D numpy array
        Vertical coordinate data e.g. pressure, height or depth for all
        instruments of the mooring
    q : 2D numpy array
        Quantity to interpolate, e.g. temperature or salinity.
    zi : float
        Interpolation vertical coordinate.
    pref : numpy array
        Polynomial coefficients describing the reference profile.
    return_diagnostics : boolean, optional
        If True, additional output is returned. Default is False.

    Return
    ------
    qi : 1D numpy array
        Interpolated data.
    ir : 1D numpy array, optional
        Row index of nearest and next nearest instrument.
    ic : 1D numpy array, optional
        Column index of nearest and next nearest instrument.
    w1 : 1D numpy array, optional
        Inverse distance weight of nearest instrument.
    w2 : 1D numpy array, optional
        Inverse distance weight of next nearest instrument.

    """

    if z.shape != q.shape:
        raise ValueError('Inputs z and q must have equal shape.')

    nr, nc = z.shape

    dz = np.abs(z - zi)
    zis = np.ones((nr,))*zi
    qrefi = np.polyval(pref, zis)

    ir = np.arange(nr)[:, np.newaxis]*np.ones((nr, 2), int)
    ic = np.argsort(dz, axis=1)[:, :2]

    zm = z[ir, ic]
    qm = q[ir, ic]
    dzc = dz[ir, ic]
    qref = np.polyval(pref, zm)

    w2 = 1./(1. + dzc[:, 1]/dzc[:, 0])
    w1 = 1. - w2

    qi1 = qrefi - qref[:, 0] + qm[:, 0]
    qi2 = qrefi - qref[:, 1] + qm[:, 1]
    qi = w1*qi1 + w2*qi2

    if return_diagnostics:
        return qi, ir, ic, w1, w2
    else:
        return qi


def interp_vel(z, u, v, sva, zi, pPsva, return_diagnostics=False):
    """Mooring velocity interpolation as described by Alex Brearley.

    It is assumed that the data consists of time series from instruments at
    various depths represented by 2D numpy arrays, A[i, j], where the index i
    determins the depth and j the time. For example,

    A = np.array([[1., 1.2, 1.3],
                  [0.5, 0.7, 0.8]])

    would describe a mooring with 2 instruments levels and each containing 3
    measurements.

    Input
    -----
    z : 2D numpy array
        Vertical coordinate data e.g. pressure, height or depth for all
        instruments of the mooring
    u : 2D numpy array
        Zonal velocity.
    v : 2D numpy array
        Meridional velocity.
    sva : 2D numpy array
        Specific volume anomaly.
    zi : float
        Interpolation vertical coordinate.
    pPsva : numpy array
        Polynomial coefficients describing the reference specific volume
        anomaly profile.
    return_diagnostics : boolean, optional
        If True, additional output is returned. Default is False.

    Return
    ------
    ui : 1D numpy array
        Interpolated zonal velocity.
    vi : 1D numpy array
        Interpolated meridional velocity.
    ir : 1D numpy array, optional
        Row index of nearest and next nearest instrument.
    ic : 1D numpy array, optional
        Column index of nearest and next nearest instrument.
    w1 : 1D numpy array, optional
        Inverse distance weight of nearest instrument.
    w2 : 1D numpy array, optional
        Inverse distance weight of next nearest instrument.

    """

    svac, ir, ic, w1, w2 = interp_quantity(z, sva, zi, pPsva, True)

    um = u[ir, ic]
    vm = v[ir, ic]
    svam = sva[ir, ic]

    theta = np.arctan2(vm[:, 0] - vm[:, 1], um[:, 0] - um[:, 1])

    ur = np.empty_like(um)
    ur[:, 0] = um[:, 0]*np.cos(theta) + vm[:, 0]*np.sin(theta)
    ur[:, 1] = um[:, 1]*np.cos(theta) + vm[:, 1]*np.sin(theta)
    vr = -um[:, 0]*np.sin(theta) + vm[:, 0]*np.cos(theta)

    sc = (ur[:, 0]*(svac - svam[:, 1]) + ur[:, 1]*(svam[:, 0] - svac))
    sc /= (svam[:, 0] - svam[:, 1])

    uc = sc*np.cos(theta) - vr*np.sin(theta)
    vc = sc*np.sin(theta) + vr*np.cos(theta)

    if return_diagnostics:
        return uc, vc, ir, ic, w1, w2
    else:
        return uc, vc


def interp1d(z, q, zi, **kwargs):
    """Wrapper around scipy.interpolate.interp1d."""
    nr, __ = z.shape
    qi = np.empty((nr,))

    for i in range(nr):
        f = ipl.interp1d(z[i, :], q[i, :], **kwargs)
        qi[i] = f(zi)

    return qi


def ll_to_xy(lon, lat):
    """Convert from longitude and latitude coordinates to Euclidian coordinates
    x, y. Only valid if the coordinates are closely spaced, i.e. dx, dy << the
    Earth's radius."""
    lllon = lon.min()
    lllat = lat.min()
    urlon = lon.max()
    urlat = lat.max()

    dlon = urlon - lllon
    dlat = urlat - lllat
    dx_ = np.squeeze(gsw.distance([lllon, urlon], [lllat, lllat]))  # dx of box
    dy_ = np.squeeze(gsw.distance([lllon, lllon], [lllat, urlat]))  # dy of box

    x = dx_*(lon - lllon)/dlon
    y = dy_*(lat - lllat)/dlat

    return x, y


def poly_area(x, y):
    """Calculate the area of a simple polygon, i.e. one that does not contain
    holes or intersections. This function uses the Shoelace formula.

    Parameters
    ----------
    x : array_like
        x coordinates.
    y : array_like
        y coordinates.

    Returns
    -------
    A : float
        Polygon area.

    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def div_vort(lon, lat, u, v):
    """Calculate spatial derivatives of velocity from mooring data using the
    Divergence theorem and Stokes theorem. The mooring positions should be
    given in metres from a common origin. At least three moorings are needed.
    The moorings should form a simple polygon, that is, one without holes or
    intersections.

    Parameters
    ----------
    lon : 1D array_like
        Longitude of moorings.
    lat : 1D array_like
        Latitude of moorings.
    u : array_like
        Zonal velocity [m s-1].
    v : array_like
        Meridional velocity [m s-1].

    Returns
    -------
    dudx : float
        Zonal gradient of zonal velocity [s-1].
    dudy : float
        Meridional gradient of zonal velocity [s-1].
    dvdx : float
        Zonal gradient of meridional velocity [s-1].
    dvdy : float
        Meridional gradient of meridional velocity [s-1].
    vort : float
        Vorticity [s-1].
    div : float
        Divergence [s-1].


    """
    x, y = ll_to_xy(lon, lat)
    xy = np.stack((x, y)).T

    # Combine into 2D arrays for compactness.
    uv = np.stack((u, v)).T

    # Define the perpendicular (dn) and parallel (ds) vectors from the polygon
    # faces.
    ds = xy - np.roll(xy, -1, axis=0)
    dn = xy - np.roll(xy, 1, axis=0)
    area = poly_area(x, y)  # Area of polygon.

    # Calculate velocity at polygon faces centres as average of moorings.
    uvc = 0.5*(np.roll(uv, -1, axis=0) + uv)

    uvn = uvc*dn
    uvs = uvc*ds

    div = np.sum(uvn)/area  # Divergence theorem.
    vort = np.sum(uvs)/area  # Stokes theorem.

    # Individual components.
    dudx = np.sum(uvn[:, 0], axis=0)/area
    dvdx = np.sum(uvs[:, 1], axis=0)/area
    dudy = -np.sum(uvs[:, 0], axis=0)/area
    dvdy = np.sum(uvn[:, 1], axis=0)/area

    return dudx, dudy, dvdx, dvdy, vort, div


def div_vort_4D(lon, lat, uv):
    """Calculate spatial derivatives of velocity from mooring data for the
    entire depth and time series at once. Pay close attention of dimensions of
    input data.

    Note
    ----
    uv = np.stack(([m['u'] for m in moorings[1:]],
                   [m['v'] for m in moorings[1:]]), axis=1)

    where u_lo is a 2D array of velocity (i, j) where i denotes time and j the
    depth level.

    Parameters
    ----------
    lon : 1D array_like
        Longitude of moorings.
    lat : 1D array_like
        Latitude of moorings.
    uv : 4D array_like
        Velocity [m s-1]. Assumes that the dimensions (i, j, k, l) are
        present where i = mooring (should go clockwise around array/sub array
        edge), j = velocity component (u, v), k = time, l = depth level.

    Returns
    -------
    dudx : float
        Zonal gradient of zonal velocity [s-1].
    dudy : float
        Meridional gradient of zonal velocity [s-1].
    dvdx : float
        Zonal gradient of meridional velocity [s-1].
    dvdy : float
        Meridional gradient of meridional velocity [s-1].
    vort : float
        Vorticity [s-1].
    div : float
        Divergence [s-1].


    """
    x, y = ll_to_xy(lon, lat)
    xy = np.stack((x, y)).T

    # Define the perpendicular (dn) and parallel (ds) vectors from the polygon
    # faces.
    ds = xy - np.roll(xy, -1, axis=0)
    dn = xy - np.roll(xy, 1, axis=0)
    area = poly_area(x, y)  # Area of polygon.

    # Calculate velocity at polygon faces centres as average of moorings.
    uvc = 0.5*(np.roll(uv, -1, axis=0) + uv)

    if np.ndim(uv) == 3:
        uvn = uvc*dn[:, :, np.newaxis]
        uvs = uvc*ds[:, :, np.newaxis]
    elif np.ndim(uv) == 4:
        uvn = uvc*dn[:, :, np.newaxis, np.newaxis]
        uvs = uvc*ds[:, :, np.newaxis, np.newaxis]

    div = np.sum(uvn, axis=(0, 1))/area  # Divergence theorem.
    vort = np.sum(uvs, axis=(0, 1))/area  # Stokes theorem.

    # Individual components.
    dudx = np.sum(uvn[:, 0, ...], axis=0)/area
    dvdx = np.sum(uvs[:, 1, ...], axis=0)/area
    dudy = -np.sum(uvs[:, 0, ...], axis=0)/area
    dvdy = np.sum(uvn[:, 1, ...], axis=0)/area

    return dudx, dudy, dvdx, dvdy, vort, div
