#!/usr/bin/env python
import sys
from xnoah import swap_coord
import xarray as xr
import os
from tqdm import tqdm
import lib.cam as lc


def load_dir(d):
    iop = xr.open_dataset(d + '/iop.nc')
    cam = xr.open_dataset(d + '/cam.nc')

    cam = cam.assign(x=cam.lon * 0 + iop.x, y=cam.lat * 0 + iop.y)

    return swap_coord(cam, {'lon': 'x', 'lat': 'y'})


def get_dirnames(files):
    return set(os.path.dirname(f) for f in files)


def main(inputs, output):
    bdate = '1999-01-01'

    dirs = get_dirnames(inputs)
    ds = xr.concat([load_dir(d) for d in tqdm(dirs)], dim='x').sortby('x')

    # common calculations
    ds = ds.rename({'lev': 'p'})
    ds['qt'] = ds.Q * 1000
    ds['sl'] = ds['T'] + ds.Z3 * 9.81 / 1004
    ds['prec'] = (ds.PRECC + ds.PRECL) * 86400 * 1000

    ds = lc.convert_dates_to_days(ds, bdate)
    ds.to_netcdf(output)


try:
    snakemake
except NameError:
    main(sys.argv[1:-1], sys.argv[-1])
else:
    main(snakemake.input, snakemake.output[0])
