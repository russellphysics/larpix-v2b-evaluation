import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit
import common



def parse_file_unique_channels(packets):
    io_group = packets['io_group'].astype(np.uint64)
    io_channel = packets['io_channel'].astype(np.uint64)
    chip_id = packets['chip_id'].astype(np.uint64)
    channel_id = packets['channel_id'].astype(np.uint64)
    unique_channels = set(common.unique_channel_id(io_group, io_channel, chip_id, channel_id))
    return unique_channels
                          


def process_bursts(adc, verbose, burst_count, skip_count):
    n_bursts = len(adc) // burst_count
    if verbose: print(n_bursts, ' burst count')
    burst_rms, burst_mean, pedestal = [[] for i in range(3)]
    for i in range(n_bursts):
        if i<skip_count: continue
        index_first=(i*burst_count)+0
        index_last=(i*burst_count)+burst_count
        burst_rms.append(np.std(adc[index_first:index_last]))
        burst_mean.append(np.mean(adc[index_first:index_last]))
        pedestal.append(adc[index_first])
    return burst_rms, burst_mean, pedestal



def fill_dictionary(packets, unique_channels, verbose, burst_count, skip_count):
    figA, axA, dA= common.fig_ax_sixty_four_channel(40,40)
#    figB, axB, dB= common.fig_ax_sixty_four_channel(40,40)
#    figC, axC, dC= common.fig_ax_sixty_four_channel(40,40)
    figD, axD, dD= common.fig_ax_sixty_four_channel(40,40)
    output = dict()
    for unique_channel in sorted(unique_channels):
        io_group_mask = packets[:]['io_group']==common.unique_2_io_group(unique_channel)
        io_channel_mask = packets[:]['io_channel']==common.unique_2_io_channel(unique_channel)
        chip_id_mask = packets[:]['chip_id']==common.unique_2_chip_id(unique_channel)
        channel_id_mask = packets[:]['channel_id']==common.unique_2_channel_id(unique_channel)
        if verbose:
            print('IO group: ',common.unique_2_io_group(unique_channel),
                  '\t IO channel: ',common.unique_2_io_channel(unique_channel),
                  '\t chip ID: ',common.unique_2_chip_id(unique_channel),
                  '\t channel ID: ',common.unique_2_channel_id(unique_channel))
        mask = np.logical_and(io_group_mask, \
                              np.logical_and(io_channel_mask, \
                                             np.logical_and(chip_id_mask, channel_id_mask)))
        
        adc=packets[mask]['dataword']
        timestamp=packets[mask]['timestamp']
        if len(adc)<2: continue
        total_rms=np.std(adc)
        if np.mean(adc)==0 or total_rms==0: continue
        burst_rms, burst_mean, pedestal = process_bursts(adc, verbose, burst_count, skip_count)
        output[unique_channel]=dict(
            load='1.2 pF',
            io_group = common.unique_2_io_group(unique_channel),
            io_channel = common.unique_2_io_channel(unique_channel),            
            chip_id = common.unique_2_chip_id(unique_channel),
            channel_id = common.unique_2_channel_id(unique_channel),
            burst_rms = burst_rms,
            burst_mean = burst_mean,
            pedestal_mean  = np.mean(pedestal),
            pedestal_rms = np.std(pedestal),
            total_rms = total_rms,
            total_rms_err = common.err_bootstrap(adc, np.std))

        channel=common.unique_2_channel_id(unique_channel)
        bins=np.linspace(60,120,61)
        axA[dA[channel][0]][dA[channel][1]].hist(adc, bins=bins)
        axA[dA[channel][0]][dA[channel][1]].legend(title='Channel '+str(int(channel)))
        axA[dA[channel][0]][dA[channel][1]].grid(True)
#        axB[dB[channel][0]][dB[channel][1]].plot(timestamp, adc, 'o')
#        axB[dB[channel][0]][dB[channel][1]].grid(True)
#        axC[dC[channel][0]][dC[channel][1]].plot(range(len(timestamp)),timestamp, 'o')
#        axC[dC[channel][0]][dC[channel][1]].grid(True)
        axD[dD[channel][0]][dD[channel][1]].hist(pedestal, bins=bins)
        axD[dD[channel][0]][dD[channel][1]].legend(title='Channel '+str(int(channel)))
        axD[dD[channel][0]][dD[channel][1]].grid(True)
        
    figA.tight_layout()
#    figB.tight_layout()
#    figC.tight_layout()
    figD.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return output



def pedestal(input_file, verbose, burst_count, skip_count):
    f = h5py.File(input_file, 'r')
    packets = common.basic_parsing(f)    
    unique_channels = parse_file_unique_channels(packets)
    return fill_dictionary(packets, unique_channels, verbose, burst_count, skip_count)


def fit_individual_channel_burst(data, mean_bins, mean_range,
                                 rms_bins, rms_range, verbose):
    output=dict()
    for unique_channel in data.keys():
        mean_vals, mean_bin_edges = np.histogram(data[unique_channel]['burst_mean'],\
                                                 bins=mean_bins, range=mean_range)
        mean_vals=list(mean_vals)
        mean_guesses = [max(mean_vals), mean_vals.index(max(mean_vals)), 5]
        mean_params, mean_cov = curve_fit(common.gauss, mean_bin_edges[:-1], mean_vals, p0=mean_guesses)
        
        rms_vals, rms_bin_edges = np.histogram(data[unique_channel]['burst_rms'],\
                                                 bins=rms_bins, range=rms_range)
        rms_vals=list(rms_vals)
        rms_guesses = [max(rms_vals), rms_vals.index(max(rms_vals)), np.std(data[unique_channel]['burst_rms'])]
        rms_params, rms_cov = curve_fit(common.gauss, rms_bin_edges[:-1], rms_vals, p0=rms_guesses)

        if verbose:
            print('IO group: ',common.unique_2_io_group(unique_channel),
                  '\t IO channel: ',common.unique_2_io_channel(unique_channel),
                  '\t chip ID: ',common.unique_2_chip_id(unique_channel),
                  '\t channel ID: ',common.unique_2_channel_id(unique_channel))
            print('BURST MEAN: ',mean_params)
            print('BURST RMS: ',rms_params)
        output[unique_channel]=dict(
            burst_mean_amplitude=mean_params[0],
            burst_mean_amplitude_err=np.sqrt(mean_cov[0][0]),
            burst_mean_mean=mean_params[1],
            burst_mean_mean_err=np.sqrt(mean_cov[1][1]),
            burst_mean_rms=mean_params[2],
            burst_mean_rms_err=np.sqrt(mean_cov[2][2]),
            burst_rms_amplitude=rms_params[0],
            burst_rms_amplitude_err=np.sqrt(rms_cov[0][0]),
            burst_rms_mean=rms_params[1],
            burst_rms_mean_err=np.sqrt(rms_cov[1][1]),
            burst_rms_rms=rms_params[2],
            burst_rms_rms_err=np.sqrt(rms_cov[2][2])
            )
    return output
        
        

def plot_individual_channel_burst(data, burst_mean_bins, burst_rms_bins, fits):
    fig0, ax0, d = common.fig_ax_sixty_four_channel(40,40)
    fig1, ax1, d = common.fig_ax_sixty_four_channel(40,40)
    for k in data.keys():
        total=data[k]['total_rms']
        total_err=data[k]['total_rms_err']
        burst_mean=data[k]['burst_mean']
        burst_rms=data[k]['burst_rms']
        channel=data[k]['channel_id']

        ax0[d[channel][0]][d[channel][1]].hist(burst_mean, bins=burst_mean_bins)
        ax1[d[channel][0]][d[channel][1]].hist(burst_rms, bins=burst_rms_bins)
        total_str=r'Total: {:.2f}$\pm${:.2f}'.format(total, total_err) 
        reset_str=r'Reset: {:.2f}$\pm${:.2f}'.format(np.std(burst_mean), \
                                                    common.err_bootstrap(burst_mean, np.median))
        ac_str=r'AC: {:.2f}$\pm${:.2f}'.format(np.mean(burst_rms), \
                                               common.err_bootstrap(burst_rms, np.median))

        ax0[d[channel][0]][d[channel][1]].legend(title='Channel '+str(channel)+'\n'+\
                                                total_str+'\n'+\
                                                reset_str+'\n'+\
                                                ac_str)
        ax0[d[channel][0]][d[channel][1]].grid(True)
        ax1[d[channel][0]][d[channel][1]].legend(title='Channel '+str(channel)+'\n'+\
                                                total_str+'\n'+\
                                                reset_str+'\n'+\
                                                ac_str)
        ax1[d[channel][0]][d[channel][1]].grid(True)
    fig0.tight_layout()
    fig1.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


    
def main(input_file, verbose, burst_count, skip_count, **kwargs):
    data = pedestal(input_file, verbose, burst_count, skip_count)
    fits = fit_individual_channel_burst(data, 100, (60,120), 100, (0,10), verbose)
    plot_individual_channel_burst(data, np.linspace(60,120,101), np.linspace(0,10,101), fits)

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', \
                        default='/home/berm/Data/larpixv2b/files/datalog_2021_09_20_12_23_20_PDT_.h5',
                        type=str, help='''Input file''')
    parser.add_argument('--verbose', default=False, type=bool, help='''Verbosity''')
    parser.add_argument('--burst_count', default=100, type=int, help='''Burst count per trigger''')
    parser.add_argument('--skip_count', default=10, type=int, help='''Burst count to disregard''')
    args = parser.parse_args()
    main(**vars(args))
