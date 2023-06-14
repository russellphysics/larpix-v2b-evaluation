import numpy as np

def unique_channel_id(io_group, io_channel, chip_id, channel_id):
    return channel_id + 100*(chip_id + 1000*(io_channel + 1000*(io_group)))


def unique_2_channel_id(unique): return unique % 100


def unique_2_chip_id(unique): return (unique//100) % 1000


def unique_2_io_channel(unique): return (unique//(100*1000)) % 1000


def unique_2_io_group(unique): return (unique//(100*1000*1000)) % 1000


def err_bootstrap(data, statistic):
    l = np.array(data)
    theta_hat_b=[]
    n_resamples=100
    for k in range(0,n_resamples):
        resampled_data=[]
        resample = np.random.choice(l, size=l.shape, replace=True)
        resampled_data.append(resample)
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    return np.std(theta_hat_b, ddof=1, axis=-1)
