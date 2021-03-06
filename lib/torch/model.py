"""Torch module for time stepping and source term estimation. This module
contains classes for performing time stepping with prescribed forcing. It also
contains a class for the Radiation and source term estimation.

"""
import logging
from collections import defaultdict

import torch
from toolz import assoc, first, valmap
from torch import nn
from .constraints import apply_linear_constraint

logger = logging.getLogger(__name__)


def _to_dict(x):
    return {
        'sl': x[..., :34],
        'qt': x[..., 34:]
    }


def _from_dict(prog):
    return torch.cat((prog['sl'], prog['qt']), -1)


def _euler_step(prog, src, h):
    for key in prog:
        x = prog[key]
        f = src[key]
        prog = assoc(prog, key, x + h * f)
    return prog


def large_scale_forcing(i, data):
    forcing = {
        key: val[i - 1]
        for key, val in data['forcing'].items()
    }
    return forcing


def compute_total_moisture(prog, data):
    w = data['constant']['w']
    return mass_integrate(prog['qt'], w)


def mass_integrate(x, w):
    return (x * w).sum(-1, keepdim=True)/1000


def compute_diagnostics(steps, lsf, w, dt):
    """Routine for computing diagnostics such as precipitation or MSE budgets
    """

    prog_start, prog_lsf, prog_nn = steps
    q_start, q_lsf, q_nn = [mass_integrate(prog['qt'], w)
                            for prog in steps]
    s_start, s_lsf, s_nn = [mass_integrate(prog['sl'], w)
                            for prog in steps]

    evap = lsf['LHF'] * 86400 / 2.51e6
    prec = evap - (q_nn - q_lsf)/dt
    return {
        'QLSF': (q_lsf - q_start)/dt/86400/1000**2,
        'QNN': (q_nn - q_lsf)/dt/86400/1000**2,
        'SLSF': 1004*(s_lsf - s_start)/dt/86400,
        'SNN': 1004*(s_nn-s_lsf)/dt/86400,
    }


def mass_integrate(x, w):
    return (x * w).sum(-1, keepdim=True)


def enforce_precip_qt(fqt, lhf, w, **kwargs):
    """Adjust moisture tendency to be positive

    .. math::

        < f >  = LHF/Lv - P

    Parameters
    ----------
    fqt : mm/day
    lhf : W/m^2
    w : kg /m^2

    """
    evap = lhf * 86400 / 2.51e6
    return apply_linear_constraint(lambda x: -mass_integrate(x, w)/1000.,
                                   -evap, fqt, inequality=True,
                                   **kwargs)


def where(cond, x, y):
    cond = cond.float()
    return cond * x + (1.0-cond) * y


def _fix_moisture(q, w, eps=1e-9):
    """Remove negative moisture points while conserving total moisture"""
    cond = q < eps
    total_moisture = mass_integrate(q, w)
    moisture_lack = mass_integrate(cond.float(), w) * eps
    moisture_valid = mass_integrate((1-cond.float()) * q, w)
    alpha = (total_moisture - moisture_lack) / moisture_valid
    return where(cond, eps, q * alpha)


def mlp(layer_sizes):
    layers = []
    n = len(layer_sizes)
    for k in range(n - 1):
        layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
        if k < n - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class RHS(nn.Module):
    def __init__(self,
                 m,
                 hidden=(),
                 scaler=None,
                 num_2d_inputs=3,
                 precip_positive=True,
                 radiation='interactive'):
        """
        Parameters
        ----------
        radiation : str
            'interactive', 'prescribed', or 'zero'.
        precip_positive : bool
            constrain precip to be positive if True
        """
        super(RHS, self).__init__()
        self.mlp = mlp((m + num_2d_inputs + m, ) + tuple(hidden) + (m, ))
        self.lin = nn.Linear(m + num_2d_inputs + m, m, bias=False)
        self.scaler = scaler
        self.bn = nn.BatchNorm1d(num_2d_inputs + m)
        self.radiation = radiation
        self.precip_positive = precip_positive
        self.num_2d_inputs = num_2d_inputs

    def forward(self, x, force, w):

        progs = x
        diags = {}
        x = self.scaler(x)
        f = self.scaler(force)

        data_2d = torch.cat((f['SHF'], f['LHF'], f['SOLIN'],
                             f['sl'], f['qt']), -1)
        data_2d = self.bn(data_2d)

        x = _from_dict(x)
        x = torch.cat((x, data_2d), -1)
        y = self.mlp(x) #+ self.lin(x)
        src = _to_dict(y)

        if self.precip_positive:
            src['qt'] = enforce_precip_qt(src['qt'], force['LHF'], w)

        return src, diags


def rhs_hidden_from_state_dict(state):
    """Determine the hidden argument for RHS from the state dictionary
    """
    import re
    pattern = re.compile('rhs\.mlp\.(\d)\.bias')
    sizes = {}
    for key, val in state.items():
        m = pattern.search(key)
        if m:
            step = int(m.group(1))
            sizes[step] = val.size(0)

    # sort the sizes by the digit
    sizes = [sizes[k] for k in sorted(sizes.keys())]

    # drop the size of the output layer
    return sizes[:-1]


class ForcedStepper(nn.Module):
    def __init__(self, rhs, h, nsteps):
        super(ForcedStepper, self).__init__()
        self.nsteps = nsteps
        self.h = h
        self.rhs = rhs

    def forward(self, data: dict):
        """

        Parameters
        ----------
        data : dict
            A dictionary containing the prognostic variables and forcing data.
        """
        data = data.copy()
        prog = data['prognostic']
        w = data['constant']['w']

        window_size = first(prog.values()).size(0)
        prog = valmap(lambda prog: prog[0], prog)
        h = self.h
        nsteps = self.nsteps

        # output array
        steps = {key: [prog[key]] for key in prog}
        # diagnostics
        diagnostics = defaultdict(list)

        for i in range(1, window_size):
            diag_step = defaultdict(lambda: 0)
            for j in range(nsteps):

                # store old state
                prog0 = prog

                # apply large scale forcings
                lsf = large_scale_forcing(i, data)
                # prog = _euler_step(prog, lsf, h / nsteps)
                prog1 = prog

                # compute and apply rhs using neural network
                src, _ = self.rhs(prog, lsf, data['constant']['w'])
                prog = _euler_step(prog, src, h / nsteps)
                prog['qt'] = _fix_moisture(prog['qt'], data['constant']['w'])
                prog2 = prog

                diags = compute_diagnostics([prog0, prog1, prog2], lsf,
                                            w=data['constant']['w'], dt=h/nsteps)

                # running average of diagnostics
                for key in diags:
                    diag_step[key] = diag_step[key] + diags[key] / nsteps

            # store accumulated diagnostics
            for key in diag_step:
                diagnostics[key].append(diag_step[key])

            # store data
            for key in prog:
                steps[key].append(prog[key])

        y = data.copy()
        y['prognostic'] = valmap(torch.stack, steps)
        y['diagnostic'] = valmap(torch.stack, diagnostics)
        return y


    @staticmethod
    def load_from_saved(d):
        from .data import scaler
        m, rhs_kw = d.pop('rhs')
        rhs_kw['scaler'] =  scaler(*rhs_kw.pop('scaler_args'))
        rhs_kw['hidden'] = rhs_hidden_from_state_dict(d['state'])
        rhs = RHS(m, **rhs_kw)


        stepper_kw = d.pop('stepper')
        stepper = ForcedStepper(rhs, **stepper_kw)
        stepper.load_state_dict(d.pop('state'))

        return stepper

    @staticmethod
    def from_file(file):
        return ForcedStepper.load_from_saved(torch.load(file))

    def to_saved(self):
        m = self.rhs.lin.out_features
        rhs_kwargs = dict(num_2d_inputs=self.rhs.num_2d_inputs,
                          precip_positive=self.rhs.precip_positive,
                          radiation=self.rhs.radiation,
                          scaler_args=self.rhs.scaler.args)
        output_dict = {
            'rhs': (m, rhs_kwargs),
            'stepper': dict(h=self.h, nsteps=self.nsteps),
            'state': self.state_dict()
        }

        return output_dict
