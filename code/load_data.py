import os
import utils

data_in = '../data'
data_out = data_in


def load_my_data(data_path=None, ret_extras=False):
    if data_path is None:
        data_path = data_in

    cc = utils.loadmat(os.path.join(data_path, 'C_raw.mat'))
    nw = utils.loadmat(os.path.join(data_path, 'NW_raw.mat'))
    ne = utils.loadmat(os.path.join(data_path, 'NE_raw.mat'))
    se = utils.loadmat(os.path.join(data_path, 'SE_raw.mat'))
    sw = utils.loadmat(os.path.join(data_path, 'SW_raw.mat'))

    moorings = [cc, nw, ne, se, sw]

    for m in moorings:
        m['tdt'] = utils.datenum_to_datetime(m['t'])

    if ret_extras:
        dt_min = 15.
        extras = {
            "dt_min": dt_min,
            "dt_sec": dt_min*60.,
            "dt_day": dt_min/1440.,
            "N_per_day": 96,
            }
        return moorings, extras
    else:
        return moorings


def load_my_data_alt(data_path=None):
    if data_path is None:
        data_path = data_in

    ca = utils.loadmat(os.path.join(data_path, 'C_alt.mat'))
    cw = utils.loadmat(os.path.join(data_path, 'C_altw.mat'))

    ca['tdt'] = utils.datenum_to_datetime(ca['t'])
    cw['tdt'] = utils.datenum_to_datetime(cw['t'])

    return ca, cw
