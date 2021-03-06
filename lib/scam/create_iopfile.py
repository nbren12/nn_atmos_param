"""See http://www.cesm.ucar.edu/models/atm-cam/docs/scam/ for a specification
of the fields contained in an IOP file.

Parameter settings for perpetual equinox
----------------------------------------

The most difficult thing to compute is the time, lat and lon which will make
the CAM radiation scheme match radiation used in SAM. I will need to set the
`solar_data_ymd` entry in CAM. See RAD_CAM/rad_full.f90:678 to see how it works
in SAM.

CAM Parameters
~~~~~~~~~~~~~~

Also see this discussion on perpetual equinox runs:
https://bb.cgd.ucar.edu/perpetual-solar-insolation-cam5

I will need to set the orbital parameters correctly. See the CAM namelist:
http://www.cesm.ucar.edu/cgi-bin/eaton/namelist/nldef2html-cam5_3

I will need to set this orbital parameters with SCAMs build-namelist.

    orb_obliq = 0.0
    orb_eccen = 0.0
    orb_mvelp = 0.0
    perpetual = .true.
    aqua_planet = .true. ! this implies that perpetual_ymd=March 21

`orb_mvelp` is the location of vernal equinox in longitude degrees, either the orb_iyear_AD must be set or the other three orb parameter must be set. default=unset

IOP File lat/lon/time
~~~~~~~~~~~~~~~~~~~~~

Because orb_mvelp = 0.0, we need the following setting in the IOPfile::

    lon = 0.0
    lat = dy*(j-ny/2)*2.5e-8 *360
    tsec = 'seconds since 1900-01-01T00:00:00'

After reading the SAM code carefully, I need to check the namelist entries for
day0, latitude0, and longitude0, do radlon and doradlat. Here is how the elements of these arrays are calculated::

    if (doradlat) then
    call task_rank_to_index(rank,it,jt)
    do j=1,ny
        latitude(:,j) = latitude0+dy*(j+jt-(ny_gl+YES3D-1)/2-1)*2.5e-8*360.
    end do
    else
    latitude(:,:) = latitude0
    end if

    if (doradlon) then
    call task_rank_to_index(rank,it,jt)
    do i=1,nx
        longitude(i,:) = longitude0+dx/cos(latitude0*pi/180.)* &
                                (i+it-nx_gl/2-1)*2.5e-8*360.
    end do
    else
    longitude(:,:) = longitude0
    end if

This means that::

    lat[j] = lat0 + dy*(j-ny/2)*2.5e-8 *360
    lon[i] = lon0 + dx*(i-nx/2)/cos(lat0 *pi/180) *i*2.5e-8 *360

2.5e-8 = 1/4e7 which is the circumference of the earth in meters. Therefore,
the factor 2.5-8 * 360 is the arclength of one degree.


Other Variables
---------------

For the SAM data, I will need to specify

- bdate
- tsec
- lev
- lat
- lon
- phis
- t
- q
- ps
- omega
- u
- v
- shflx
- lhflk
- divT
- vertdivT
- divq
- vertdivq

"""
import os
import itertools
import numpy as np
import xarray as xr
from toolz import curry, valmap

from xnoah import swap_coord
from ..thermo import omega_from_w
from ..advection import vertical_advection, horizontal_advection


circumference_earth = 4.0075e7


def open_and_merge(file_2d, file_3d, stat_file):
    data_3d = xr.open_dataset(file_3d).drop('p')
    data_2d = xr.open_dataset(file_2d)
    data_2d = data_2d.isel(time=np.argsort(data_2d.time.values))
    stat = xr.open_dataset(stat_file)

    # need to upsampel stat
    stat = stat.sel(time=data_3d.time, method='nearest')\
               .assign(time=data_3d.time)

    return xr.merge((data_3d,
                     stat.p,
                     stat.RHO[-1],
                     data_2d.compute(),
                     stat.Ps),
                    join='inner')


def compute_divq(u, v, qv):
    tend = -horizontal_advection(u, v, qv)
    tend = tend * (qv > .001)
    return tend

def compute_tendencies(data):

    # this needs to include the w g/cp term in the temperature budget so use
    # static energy
    s  = 9.81/1004 * data.z + data.TABS
    divT3d = -(vertical_advection(data.W, s) +
               horizontal_advection(data.U, data.V, s))

    # moisture tendencies
    # need to use another name for this array
    Q_dten = -(horizontal_advection(data.U, data.V, data.QV) +
               vertical_advection(data.W, data.QV))/1000

    return dict(
        divT3d=divT3d,
        Q_dten=Q_dten
    )


def x_to_lon(x):
    return xr.DataArray(x/circumference_earth*360,
                        attrs={'units': 'deg E',
                               'long_name': 'longitude'})


def y_to_lat(y):
    n = len(y)
    if n % 2 == 1:
        raise ValueError("Y coordinate must have even number of points")
    else:
        n2 = n//2
    ymid = (y[n2-1] + y[n2])/2
    lat = (y-ymid)/circumference_earth*360
    return xr.DataArray(lat,
                        attrs={'units': 'deg N',
                               'long_name': 'latitude'})

vars_3d = ['q', 'T', 'u', 'v', 'divT', 'divq']
vars_2d = ['shflk', 'lhflx', 'Ptend', 'Ps', 'phis']

def expand_dims(x):
    if x.name in vars_3d + vars_2d:
        return x.expand_dims(['lat', 'lon'])
    else:
        return x


def prepare_iop_dataset(data):
    data_vars = dict(
        # time related variables
        bdate=xr.DataArray(990101, attrs={'units': "yymmdd"}),
        tsec=xr.DataArray(data.time*86400, attrs={'units': "s"}),
        # spatial coordinates
        lev=xr.DataArray(data.p*100, attrs={'units': 'Pa'}),
        lat=y_to_lat(data.y),
        lon=x_to_lon(data.x),
        # surface variables
        phis=0.0,
        Ps=xr.DataArray(data.Ps*100, attrs={'units': 'Pa'}),
        Ptend=data.Ps * 0,
        shflx=data.SHF,
        lhflx=data.LHF,
        # 3d variables
        q=xr.DataArray(data.QV/1000, attrs={'units': 'kg/kg'}),
        T=data.TABS,
        u=data.U,
        v=data.V,
        omega=xr.DataArray(omega_from_w(data.W, data.RHO[-1]),
                           attrs={'units': 'Pa/s'}),
        # diagnostics
        SOLIN=data.SOLIN,
        QRAD=data.QRAD,
        PREC=data.Prec,
        SST=data.SST,
    )

    data_vars.update(compute_tendencies(data))

    ds = xr.Dataset(data_vars)
    ds = (swap_coord(ds, {"time": "tsec", 'x': 'lon', 'y': 'lat', 'z': 'lev'})
          .transpose('tsec', 'lev', 'lat', 'lon')
          .sortby('lev'))

    return ds.drop('time')


def prepare_namelist(loc):
    namelist_template = """
&atm
    iopfile='iop.nc'
    nhtfrq=-3
    single_column=.true.
    scm_iop_srf_prop = .true.
    scmlat= {lat:.4f}
    scmlon= 0.0
    aqua_planet = .true.
    orb_obliq = 0.0
    orb_eccen = 0.0
    orb_mvelp = 0.0
    orb_mode = 'fixed_parameters'
    perpetual = .true.
    start_tod = {start_tod}
    stop_tod = {stop_tod}
    stop_n = {stop_n}
    fincl1 = 'DIVQ3D:A'
/
    """

    format_params = dict(
        lat=float(loc.lat),
        lon=float(loc.lon))

    format_params.update(start_stop_params(loc.tsec))
    return namelist_template.format(**format_params)


def start_stop_params(tsec):
    val = dict(start_tod=tsec[0] % 86400,
               stop_tod=tsec[-1] % 86400,
               stop_n=(tsec[-1]-tsec[0])//86400)

    return valmap(int, val)


def save_iop_dir(dirname, loc):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    output_nc = os.path.join(dirname, "iop.nc")
    output_nml = os.path.join(dirname, "namelist.txt")
    loc = loc.apply(expand_dims)\
             .transpose('tsec', 'lev', 'lat', 'lon')
    # for some reason SCAM dies when lon = 0
    # something to do with initializing the land vegetation array
    loc.lon[0] = 0.0
    loc.to_netcdf(output_nc)
    with open(output_nml, "w") as f:
        f.write(prepare_namelist(loc))

    return dirname


def save_all_dirs(iop, output_dir):
    ij = itertools.product(range(len(iop.lon)), range(len(iop.lat)))
    for i, j in ij:
        print(f"Saving {i}-{j}")
        dirname = os.path.join(output_dir, f"{i}-{j}")
        loc = iop.isel(lon=i, lat=j)
        save_iop_dir(dirname, loc)


def main(file_2d, files_3d, stat, output):
    # file_2d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc"
    # files_3d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/*.nc"
    # stat = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"

    data = open_and_merge(file_2d, files_3d, stat)
    iop = prepare_iop_dataset(data).compute().isel(lat=slice(24, 40))
    iop.to_netcdf(output)


