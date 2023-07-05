import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit
import common


jan23cryo_digital_gain={
    (0,10,40):['datalog_2022_01_23_15_53_57_PST_.h5',
               'datalog_2022_01_23_19_05_49_PST_.h5',
               'datalog_2022_01_23_19_09_56_PST_.h5',
               'datalog_2022_01_23_19_47_39_PST_.h5'],
    (0,25,45):['datalog_2022_01_23_15_48_20_PST_.h5',
               'datalog_2022_01_23_19_04_34_PST_.h5',
               'datalog_2022_01_23_19_12_29_PST_.h5',
               'datalog_2022_01_23_19_46_20_PST_.h5'],
    (0,50,50):['datalog_2022_01_23_15_39_20_PST_.h5',
               'datalog_2022_01_23_19_01_58_PST_.h5',
               'datalog_2022_01_23_19_13_42_PST_.h5',
               'datalog_2022_01_23_19_40_12_PST_.h5'],
    (0,100,50):['datalog_2022_01_23_15_58_21_PST_.h5',
                'datalog_2022_01_23_19_00_26_PST_.h5',
                'datalog_2022_01_23_19_16_00_PST_.h5',
                'datalog_2022_01_23_19_38_55_PST_.h5'],
    (0,200,50):['datalog_2022_01_23_17_56_43_PST_.h5',
                'datalog_2022_01_23_18_57_59_PST_.h5',
                'datalog_2022_01_23_19_17_19_PST_.h5',
                'datalog_2022_01_23_19_36_36_PST_.h5'],
    (0,400,100):['datalog_2022_01_23_18_01_26_PST_.h5',
                 'datalog_2022_01_23_18_55_21_PST_.h5',
                 'datalog_2022_01_23_19_20_13_PST_.h5',
                 'datalog_2022_01_23_19_34_09_PST_.h5'],
    (0,450,100):['datalog_2022_01_23_18_06_58_PST_.h5',
                 'datalog_2022_01_23_18_54_03_PST_.h5',
                 'datalog_2022_01_23_19_22_34_PST_.h5',
                 'datalog_2022_01_23_19_32_50_PST_.h5'],
    (0,500,100):['datalog_2022_01_23_14_36_10_PST_.h5'],
    (1,10,35):['datalog_2022_01_23_15_56_06_PST_.h5',
               'datalog_2022_01_23_19_06_56_PST_.h5',
               'datalog_2022_01_23_19_48_48_PST_.h5',
               'datalog_2022_01_23_19_11_17_PST_.h5'],
    (1,25,37):['datalog_2022_01_23_15_50_30_PST_.h5'],
    (1,50,40):['datalog_2022_01_23_15_45_05_PST_.h5',
               'datalog_2022_01_23_19_03_03_PST_.h5',
               'datalog_2022_01_23_19_14_47_PST_.h5',
               'datalog_2022_01_23_19_41_20_PST_.h5'],
    (1,100,50):['datalog_2022_01_23_16_00_59_PST_.h5'],
    (1,200,50):['datalog_2022_01_23_17_59_06_PST_.h5',
                'datalog_2022_01_23_18_59_05_PST_.h5',
                'datalog_2022_01_23_19_18_41_PST_.h5',
                'datalog_2022_01_23_19_37_39_PST_.h5'],
    (1,400,100):['datalog_2022_01_23_18_04_43_PST_.h5',
                 'datalog_2022_01_23_18_56_25_PST_.h5',
                 'datalog_2022_01_23_19_21_19_PST_.h5',
                 'datalog_2022_01_23_19_35_15_PST_.h5'],
    (1,800,100):['datalog_2022_01_23_18_09_14_PST_.h5',
                 'datalog_2022_01_23_18_48_08_PST_.h5',
                 'datalog_2022_01_23_19_24_06_PST_.h5',
                 'datalog_2022_01_23_19_31_27_PST_.h5'],
    (1,900,100):['datalog_2022_01_23_18_11_27_PST_.h5',
                 'datalog_2022_01_23_18_45_54_PST_.h5',
                 'datalog_2022_01_23_19_25_18_PST_.h5',
                 'datalog_2022_01_23_19_30_10_PST_.h5']
}



def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))



def fit_mean(adc, ax, row, adc_cut, rms_cut, show_adc, verbose):
#    guesses=[max(vals[5:]), vals[5:].index(max(vals[5:])), 1]
#    params, cov = curve_fit(gauss, bin_edges[5:-1], vals[5:], \
#                            p0=guesses)
#    ax[row].plot(bin_edges[5:-1], gauss(bin_edges[5:-1], *params), '-')

    adc=[i for i in adc if i > 5 and i<adc_cut]
    vals, bin_edges = np.histogram(adc, bins=256, range=(0,256))
    vals=list(vals)
    guesses=[max(vals), vals.index(max(vals)), 1]
    params, cov = curve_fit(gauss, bin_edges[:-1], vals, \
                            p0=guesses)
    ax[row].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), '-')
    if verbose:
        print('--------MODE-------')
        print('amplitude: ',params[0],\
              '\t delta_amplitude: ',np.sqrt(cov[0][0]))
        print('mu: ',params[1],'\t delta_mu: ',np.sqrt(cov[1][1]))
        print('sigma: ',params[2],'\t delta_sigma: ',np.sqrt(cov[2][2]))
        print('\n')

        
def successive_adc_fit(adc, ax, row, initial_adc_cut, rms_cut, \
                       show_adc, verbose):
    out=[]
    adc=[i for i in adc if i>=initial_adc_cut]
    vals, bin_edges = np.histogram(adc, bins=256, range=(0,256))
    vals=list(vals)
    guesses=[max(vals), vals.index(max(vals)), 1]
    if verbose: print('GUESSES: ', guesses)
    params, cov = curve_fit(gauss, bin_edges[:-1], vals, \
                            p0=guesses)
    if verbose:
        print('amplitude: ',params[0],\
              '\t delta_amplitude: ',np.sqrt(cov[0][0]))
        print('mu: ',params[1],'\t delta_mu: ',np.sqrt(cov[1][1]))
        print('sigma: ',params[2],'\t delta_sigma: ',np.sqrt(cov[2][2]))
    ax[row].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), '-')    
    result=(params[1], np.sqrt(cov[1][1]),
            params[2], np.sqrt(cov[2][2]))
    out.append(result)

    adc=[i for i in adc if i>=params[1]+params[2] or i<=params[1]-params[2]]
    vals, bin_edges = np.histogram(adc, bins=256, range=(0,256))
    vals=list(vals)
    guesses=[max(vals), vals.index(max(vals)), 1]
    if verbose: print('GUESSES: ', guesses)
    params, cov = curve_fit(gauss, bin_edges[:-1], vals, \
                            p0=guesses)
    if verbose:
        print('amplitude: ',params[0],\
              '\t delta_amplitude: ',np.sqrt(cov[0][0]))
        print('mu: ',params[1],'\t delta_mu: ',np.sqrt(cov[1][1]))
        print('sigma: ',params[2],'\t delta_sigma: ',np.sqrt(cov[2][2]))
    ax[row].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), '-')    
    result=(params[1], np.sqrt(cov[1][1]),
            params[2], np.sqrt(cov[2][2]))
    out.append(result)

    adc=[i for i in adc if i>=params[1]+params[2] or i<=params[1]-params[2]]
    vals, bin_edges = np.histogram(adc, bins=256, range=(0,256))
    vals=list(vals)
    guesses=[max(vals), vals.index(max(vals)), 1]
    if verbose: print('GUESSES: ', guesses)
    params, cov = curve_fit(gauss, bin_edges[:-1], vals, \
                            p0=guesses)
    if verbose:
        print('amplitude: ',params[0],\
              '\t delta_amplitude: ',np.sqrt(cov[0][0]))
        print('mu: ',params[1],'\t delta_mu: ',np.sqrt(cov[1][1]))
        print('sigma: ',params[2],'\t delta_sigma: ',np.sqrt(cov[2][2]))
    ax[row].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), '-')    
    result=(params[1], np.sqrt(cov[1][1]),
            params[2], np.sqrt(cov[2][2]))
    out.append(result)
    
        
def fit_adc_response(adc, ax, row, adc_cut, rms_cut, show_adc, verbose):
    do_fit=True
    fit_mean(adc, ax, row, adc_cut, rms_cut, show_adc, verbose)
    revised_adc=[i for i in adc if i > adc_cut]
    vals, bin_edges = np.histogram(revised_adc, bins=256, range=(0,256))
    guesses=[max(vals),np.mean(revised_adc),1]
    if verbose: print('GUESSES: ',guesses,'\n rms: ',np.std(revised_adc))
    if np.std(revised_adc)>=rms_cut:
        params, cov = curve_fit(gauss, bin_edges[:-1], vals, \
                                p0=guesses)
        if show_adc:
            ax[row].plot(bin_edges, gauss(bin_edges, *params), '-')
        if verbose:
            print('amplitude: ',params[0],\
                  '\t delta_amplitude: ',np.sqrt(cov[0][0]))
            print('mu: ',params[1],'\t delta_mu: ',np.sqrt(cov[1][1]))
            print('sigma: ',params[2],'\t delta_sigma: ',np.sqrt(cov[2][2]))
            print('\n\n\n')
            return (params[1], np.sqrt(cov[1][1]),
                    params[2], np.sqrt(cov[2][2]))
    else:
        return (np.mean(revised_adc), 0,
                np.std(revised_adc), 0)



def plot_adc(p, title, row, adc_cut, rms_cut, d, gain, input_mV, \
             ax, show_adc, verbose, single_channel):
    unique_channels = np.unique(p['channel_id'])
    for uc in unique_channels:
        if single_channel>=0:
            if single_channel!=uc: continue
        mask=p['channel_id']==uc
        adc=p[mask]['dataword']
        if show_adc:
            ax[row].hist(adc, bins=np.linspace(0,255,256), \
                         alpha=0.5, label=str(uc))
            ax[row].axvline(x=adc_cut, linestyle='--', color='r')
        if verbose: print('gain: ',gain,' input mV: ',input_mV,' channel: ',uc)
        successive_adc_fit(adc, ax, row, adc_cut, rms_cut, \
                       show_adc, verbose)
        d[(gain, input_mV, uc)]=(-1,-1,-1,-1)
#        fit = fit_adc_response(adc, ax, row, adc_cut, rms_cut, \
#                               show_adc, verbose)
#        d[(gain, input_mV, uc)]=fit




def format_adc(fig, fig1, ax, ax1, nrows):
    for i in range(len(nrows)):
        for j in range(nrows[i]):
            if i==0:
                ax[nrows[i]-1].set_xlabel('ADC')
                ax[j].set_ylabel('Trigger Count')
                ax[j].grid(True)
                ax[j].set_xlim(0,255)
                #ax[j].legend(title='Channel')
            if i==1:
                ax1[nrows[i]-1].set_xlabel('ADC')
                ax1[j].set_ylabel('Trigger Count')
                ax1[j].grid(True)
                ax1[j].set_xlim(0,255)
                #ax1[j].legend(title='Channel')
        if i==0:
            fig.suptitle(r'4$\mu$V/e$^-$ Configured Gain')
            #fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)
        if i==1:
            fig1.suptitle(r'2$\mu$V/e$^-$ Configured Gain')
            #fig1.tight_layout()
            fig1.subplots_adjust(wspace=0, hspace=0)
    plt.show()



def format_fit(d, show_fit, cutoff):
    if show_fit: fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    out=dict()
    channels = set([k[2] for k in d.keys()])
    gains = set([k[0] for k in d.keys()])
    for g in gains:
        injected = list(set([k[1] for k in d.keys() if k[0]==g]))
        injected.sort()
        for c in channels:
            x=[k[1] for k in d.keys() if k[0]==g and k[2]==c]
            y=[d[k][0] for k in d.keys() if k[0]==g and k[2]==c]
            err=[d[k][1] for k in d.keys() if k[0]==g and k[2]==c]
            if show_fit:
                ax1[g].errorbar(x, y, yerr=err, fmt='o', label=str(c))
            modx=[k[1] for k in d.keys() \
                  if k[0]==g and k[2]==c and k[1] in injected[:cutoff[g]]]
            mody=[d[k][0] for k in d.keys() \
                  if k[0]==g and k[2]==c and k[1] in injected[:cutoff[g]]]
            coef=np.polyfit(modx,mody,1)
            poly1d_fn = np.poly1d(coef)
            out[(g,c)]=(coef[0], coef[1])
            if show_fit: ax1[g].plot(modx, poly1d_fn(modx), '--')
    if show_fit:
        for i in range(2):
            ax1[i].set_xlabel('Injected Pulse [mV]')
            ax1[i].set_ylabel('Response [ADC]')
            ax1[i].set_ylim(0,255)
            ax1[i].grid(True)
            ax1[i].legend(title='Channel')
        plt.show()
    return out



def format_response(linear_fit):
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    gains=set([k[0] for k in linear_fit.keys()])
    r=dict()
    for g in gains:
        if g not in r.keys(): r[g]=[]
    for k in linear_fit.keys(): r[k[0]].append(linear_fit[k][0])
    for k in r.keys():
        if k==0: bins=np.linspace(0.4,0.5,21)
        if k==1: bins=np.linspace(0.2,0.3,21)
        ax2[k].hist(r[k], bins=bins)
        ax2[k].grid(True)
        ax2[k].set_xlabel('ADC / mV')
        ax2[k].set_ylabel('Channel Count')
    plt.show()

    

def main(input_dict, file_path, show_adc, show_fit, show_response, \
         verbose, single_channel, rms_cut, **kwargs):
    d=dict()
    nrows=[0,0]; nrow_ctr=[0,0]
    input_d = input_dict
    for k in input_d.keys():
        if k[0]==0: nrows[0]+=1
        if k[0]==1: nrows[1]+=1

    fig,ax,fig0,ax0=[None for i in range(4)]
    if show_adc:
        fig, ax = plt.subplots(nrows=nrows[0], ncols=1, \
                               sharex=True, sharey=True,\
                               figsize=(24,24))
        fig0, ax0 = plt.subplots(nrows=nrows[1], ncols=1, \
                                 sharex=True, sharey=True\
                                 figsize=(24,24))

    for k in input_d.keys():
        for f in input_d[k]:
            if verbose: print(f)
            fin = h5py.File(file_path+f,'r')
            p = common.basic_parsing(fin)
            
            if k[0]==0:
                #plot_adc(p, str(k[1])+' mV', nrow_ctr[k[0]], k[2], rms_cut, \
                #         d, k[0], k[1], ax, show_adc, verbose, single_channel)
                plot_adc(p, str(k[1])+' mV', nrow_ctr[k[0]], 5, rms_cut, \
                         d, k[0], k[1], ax, show_adc, verbose, single_channel)
            if k[0]==1:
                #plot_adc(p, str(k[1])+' mV', nrow_ctr[k[0]], k[2], rms_cut, \
                #         d, k[0], k[1], ax0, show_adc, verbose, single_channel)
                plot_adc(p, str(k[1])+' mV', nrow_ctr[k[0]], 5, rms_cut, \
                         d, k[0], k[1], ax0, show_adc, verbose, single_channel)
        nrow_ctr[k[0]]+=1

    if show_adc: format_adc(fig, fig0, ax, ax0, nrows)
    linear_fit = format_fit(d, show_fit, cutoff=[-2,-1])
    if show_response: format_response(linear_fit)
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dict', default=jan23cryo_digital_gain,
                        type=str, help='''Input dictionary string ID''')
    parser.add_argument('--file_path',
                        default='/home/russell/LArPix/v2_ASIC_paper/data/23january2022/gain_digital_linearity/',
                        type=str, help='''File directory''')
    parser.add_argument('--single_channel', default=-1, type=int, \
                        help='''Single channel to plot''')
    parser.add_argument('--show_adc',
                        default=False, type=bool, \
                        help='''Plot 1D ADC histogram and fit''')
    parser.add_argument('--show_fit',
                        default=False, type=bool, \
                        help='''Plot fit results''')
    parser.add_argument('--show_response',
                        default=False, type=bool, \
                        help='''Plot extracted response''')
    parser.add_argument('--verbose', default=False, type=bool, \
                        help='''Verbosity''')
    parser.add_argument('--rms_cut', default=0.1, type=float, \
                        help='''ADC RMS threshold to perform fit''')
    args = parser.parse_args()
    main(**vars(args))
