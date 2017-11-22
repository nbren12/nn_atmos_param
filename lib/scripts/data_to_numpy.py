import numpy as np
from sklearn.externals import joblib
from xnoah.data_matrix import stack_cat
from lib.util import compute_weighted_scale, weights_to_np, scales_to_np

import xarray as xr


def prepvar(X, feature_dims=['z'], sample_dims=['time', 'x', 'y']):
    # select only the tropics
    return stack_cat(X, "features", ['z'])


def prepare_data(input_files, forcing_files, weight_file):

    # load the data
    inputs = xr.open_mfdataset(input_files)
    forcings = xr.open_mfdataset(forcing_files)

    # only use tropics
    inputs = inputs.isel(y=slice(24, 40))
    forcings = forcings.isel(y=slice(24, 40))

    # get weights
    w = xr.open_dataarray(weight_file)

    # compute scales
    sample_dims = ['x', 'y', 'time']
    scales = compute_weighted_scale(w, sample_dims=sample_dims,
                                    ds=inputs)

    # stack the features
    X = prepvar(inputs)
    G = prepvar(forcings)

    scales_np = scales_to_np(scales, X.indexes['features'])
    w_np = weights_to_np(w, X.indexes['features'])

    return {
        'X': np.asarray(X),
        'G': np.asarray(G),
        'scales': scales_np,
        'w': w_np
    }



def main():
    input_files = snakemake.input.inputs
    forcing_files = snakemake.input.forcing
    weight_file = snakemake.input.weight

    output_data = prepare_data(input_files, forcing_files, weight_file)

    np.savez(snakemake.output[0], **output_data)

if __name__ == '__main__':
    main()

# forcing_files = [
#     "data/calc/forcing/ngaqua/sl.nc",
#     "data/calc/forcing/ngaqua/qt.nc",
# ]

# input_files = ["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"]

# weight_file = "data/processed/ngaqua/w.nc"


# output_data = prepare_data(input_files, forcing_files, weight_file)
# from IPython import embed; embed()
