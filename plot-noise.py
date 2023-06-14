import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import common

rt_pedestal='/home/brussell/larpix-v2b/cts/data/pedestal/operation_pedestal/Jan18_rt/datalog_2022_01_18_15_55_51_PST_.h5'

cryo_pedestal='/home/brussell/larpix-v2b/cts/data/pedestal/operation_pedestal/Jan18/datalog_2022_01_18_14_48_07_PST_.h5'

label_dict={'mean':'Pedestal Mean [ADC]', 'std':'Pedestal RMS [ADC]'}
figure_dict={'mean':'pedestal_mean', 'std':'pedestal_rms'}

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


def simple_dict(f):
    f = h5py.File(f,'r')
    data_mask = f['packets'][:]['packet_type']==0
    valid_parity_mask = f['packets'][:]['valid_parity']==1
    mask = np.logical_and(data_mask, valid_parity_mask)
    data = f['packets'][mask]
    io_group = data['io_group'].astype(np.uint64)
    io_channel = data['io_channel'].astype(np.uint64)
    chip_id = data['chip_id'].astype(np.uint64)
    channel_id = data['channel_id'].astype(np.uint64)
    unique_channels = set(common.unique_channel_id(io_group, io_channel, chip_id, channel_id))

    d = dict()
    for uc in sorted(unique_channels):
        channel_mask = common.unique_channel_id(io_group, io_channel, chip_id, channel_id) == uc
        adc = data[channel_mask]['dataword']
        d[common.unique_2_channel_id(uc)] = dict(
            io_group = common.unique_2_io_group(uc),
            mean = np.mean(adc),
            mean_err = common.err_bootstrap(adc, np.mean),
            std = np.std(adc),
            std_err = common.err_bootstrap(adc, np.std)
            )
    return d

def plot_histogram(r, c, metric, lowbin, highbin, nbins):
    plt.figure(figure_dict[metric])
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist([r[key][metric] for key in r.keys() if r[key]['io_group']==1], bins=np.linspace(lowbin,highbin,nbins), alpha=0.5, label='RT A')
    ax[0].hist([c[key][metric] for key in c.keys() if c[key]['io_group']==1], bins=np.linspace(lowbin,highbin,nbins), alpha=0.5, label='Cryo A')
    ax[1].hist([r[key][metric] for key in r.keys() if r[key]['io_group']==2], bins=np.linspace(lowbin,highbin,nbins), alpha=0.5, label='RT C')
    ax[1].hist([c[key][metric] for key in r.keys() if r[key]['io_group']==2], bins=np.linspace(lowbin,highbin,nbins), alpha=0.5, label='Cryo C')

    for i in range(2):
        ax[i].set_xlabel(label_dict[metric])
        ax[i].set_ylabel('Channel Count')
        ax[i].grid(True)
        ax[i].legend()
    
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(figure_dict[metric]+'.png')
    
def main():
    rt = simple_dict(rt_pedestal)
    cryo = simple_dict(cryo_pedestal)

    plot_histogram(rt, cryo, 'mean', 0, 255, 256)
    plot_histogram(rt, cryo, 'std', 0, 255, 256)

if __name__=='__main__':
    main()
