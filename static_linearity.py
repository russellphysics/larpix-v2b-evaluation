import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit



fig_dict={0:(0,0),1:(0,1),2:(0,2),3:(0,3),4:(0,4),5:(0,5),6:(0,6),7:(0,7),
        8:(1,0),9:(1,1),10:(1,2),11:(1,3),12:(1,4),13:(1,5),14:(1,6),15:(1,7),
        16:(2,0),17:(2,1),18:(2,2),19:(2,3),20:(2,4),21:(2,5),15:(2,6),16:(2,7),
        24:(3,0),25:(3,1),26:(3,2),27:(3,3),28:(3,4),29:(3,5),30:(3,6),31:(3,7),
        32:(4,0),33:(4,1),34:(4,2),35:(4,3),36:(4,4),37:(4,5),38:(4,6),39:(4,7),
        40:(5,0),41:(5,1),42:(5,2),43:(5,3),44:(5,4),45:(5,5),46:(5,6),47:(5,7),
        48:(6,0),49:(6,1),50:(6,2),51:(6,3),52:(6,4),53:(6,5),54:(6,6),55:(6,7),
        56:(7,0),57:(7,1),58:(7,2),59:(7,3),60:(7,4),61:(7,5),62:(7,6),63:(7,7)}
ch_dict={7:(0,0),11:(0,1),15:(0,2),23:(0,3),
         27:(1,0),30:(1,1),39:(1,2),43:(1,3),
         45:(2,0),55:(2,1),59:(2,2),60:(2,3)}



def gauss(x, A, mu, sigma): return A*np.exp(-(x-mu)**2/(2*sigma**2))



def basic_parsing(f):
    parity_mask=f['packets'][:]['valid_parity']==1
    data_mask=f['packets'][:]['packet_type']==0
    mask = np.logical_and(parity_mask,data_mask)
    return f['packets'][mask]



def plot_adc_dist(p, nrows=3, ncols=4):
    unique_channels = np.unique(p['channel_id'])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24,12))
    bins=np.linspace(0,255,256)
    for uc in unique_channels:
        mask = p['channel_id']==uc
        adc=p[mask]['dataword']
        ax[ch_dict[uc][0]][ch_dict[uc][1]].hist(adc,bins=bins)
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_xlabel('ADC')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_ylabel('Packet Count')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_yscale('log')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_title('Channel '+str(uc))
    plt.tight_layout()
    plt.show()
    

def plot_timestamp_vs_adc(p, nrows=3, ncols=4):
    unique_channels = np.unique(p['channel_id'])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24,12))
    for uc in unique_channels:
        mask = p['channel_id']==uc
        adc=p[mask]['dataword']
        timestamp = p[mask]['timestamp']
        ax[ch_dict[uc][0]][ch_dict[uc][1]].scatter(timestamp,adc)
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_xlabel('Timestamp')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_ylabel('ADC')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_title('Channel '+str(uc))
    plt.tight_layout()
    plt.show()

    

def find_extrema(l, out):
    temp = max(l)
    if abs(min(l))>temp: temp=abs(min(l))
    out.append(temp)
    return 
    


def find_static_linearity(p, nrows=3, ncols=4):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(30,22))
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    dnl_extrema, inl_extrema = [[] for i in range(2)]
    unique_channels = np.unique(p['channel_id'])
    for uc in unique_channels:
        mask = p['channel_id']==uc
        adc=p[mask]['dataword']
        vals, bin_edges = np.histogram(adc, bins=256, range=(0,256))

        meanDict, runningSum = [{} for k in range(2)]
        dnl_val, dnl_adc, inl_val = [{} for k in range(3)]
        for k in range(1,255): # exclude first and last ADC bins
            meanDict[k]=0; runningSum[k]=0
            dnl_val[k]=[]; dnl_adc[k]=[]; inl_val[k]=[]

        for k in range(1,255): # exclude first and last ADC bins
            for l in range(1,255):
                if l>=k: meanDict[l] += vals[k]

        for key in meanDict.keys(): meanDict[key] = meanDict[key]/key

        for k in range(1,255):
            for l in range(1,255):
                if l>=k:
                    dnl = (vals[k]-meanDict[l])/meanDict[l]
                    dnl_val[l].append( dnl )
                    runningSum[l]+=dnl
                    inl_val[l].append( runningSum[l] )
                    dnl_adc[l].append(k)
                    
        find_extrema(dnl_val[254], dnl_extrema)
        find_extrema(inl_val[254], inl_extrema)
        
        ax[ch_dict[uc][0]][ch_dict[uc][1]].plot(dnl_adc[254],dnl_val[254])
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_xlabel('ADC')
        if ch_dict[uc][1]==0:
            ax[ch_dict[uc][0]][ch_dict[uc][1]].set_ylabel('DNL [LSB]', \
                                                          color='tab:blue', \
                                                          fontweight='bold')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].legend(title='Channel '+str(uc), loc='upper center')
        ax[ch_dict[uc][0]][ch_dict[uc][1]].grid(True)
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_xlim(0,255)
        ax[ch_dict[uc][0]][ch_dict[uc][1]].set_ylim(-0.5,0.5)
        axtwin = ax[ch_dict[uc][0]][ch_dict[uc][1]].twinx()
        axtwin.plot(dnl_adc[254],inl_val[254], color='k')
        if ch_dict[uc][1]==3:
            axtwin.set_ylabel('INL [LSB]', fontweight='bold')
        axtwin.set_ylim(-4,4)
        if ch_dict[uc][1]!=3:
            axtwin.set_yticklabels([])
        if uc==7:     
            ax2[0].plot(dnl_adc[254], dnl_val[254])
            ax3[0].plot(dnl_adc[254], inl_val[254])

    fig.subplots_adjust(wspace=0, hspace=0)
#    fig.tight_layout()
    bins=np.linspace(0.2,0.4,21)
    hist, bin_edges = np.histogram(dnl_extrema, bins=20, \
                                   range=(0.2,0.4))
    params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                            p0=[max(hist), np.mean(dnl_extrema),
                                np.std(dnl_extrema)])
    ax1[0].hist(dnl_extrema, bins=bins)
    ax1[0].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), \
                '-', label=r'Fit'+'\n'+'$\mu$={:.2f}$\pm${:.2f}'.format(params[1], np.sqrt(cov[1][1]))+'\n'+'$\sigma$={:.2f}$\pm${:.2f}'.format(params[2], np.sqrt(cov[2][2])))
    
    ax1[0].set_xlabel('DNL Extrema')
    ax1[0].set_ylabel('Channel Count / 0.01 LSB')
    ax1[0].grid(True)
    ax1[0].set_xlim(0.2,0.4)
    ax1[0].legend()

    ax2[1].hist(dnl_extrema, bins=bins)
    ax2[1].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), \
                '-', label=r'Fit'+'\n'+'$\mu$={:.2f}$\pm${:.2f}'.format(params[1], np.sqrt(cov[1][1]))+'\n'+'$\sigma$={:.2f}$\pm${:.2f}'.format(params[2], np.sqrt(cov[2][2])))

    bins=np.linspace(2,4,21)
    hist, bin_edges = np.histogram(inl_extrema, bins=20, \
                                   range=(2,4))
    params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                            p0=[max(hist), np.mean(inl_extrema),
                                np.std(inl_extrema)])
    ax1[1].hist(inl_extrema, bins=bins)
    ax1[1].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), \
                '-', label=r'Fit'+'\n'+'$\mu$={:.2f}$\pm${:.2f}'.format(params[1], np.sqrt(cov[1][1]))+'\n'+'$\sigma$={:.2f}$\pm${:.2f}'.format(params[2], np.sqrt(cov[2][2])))
    ax1[1].set_xlabel('INL Extrema')
    ax1[1].set_ylabel('Channel Count / 0.1 LSB')
    ax1[1].grid(True)
    ax1[1].set_xlim(2,4)
    ax1[1].legend()

    ax3[1].hist(inl_extrema, bins=bins)
    ax3[1].plot(bin_edges[:-1], gauss(bin_edges[:-1], *params), \
                '-', label=r'Fit'+'\n'+'$\mu$={:.2f}$\pm${:.2f}'.format(params[1], np.sqrt(cov[1][1]))+'\n'+'$\sigma$={:.2f}$\pm${:.2f}'.format(params[2], np.sqrt(cov[2][2])))
#    fig1.tight_layout()

    for i in range(2):
        ax2[i].grid(True)
        ax3[i].grid(True)
    ax2[0].set_xlabel('ADC')
    ax2[0].set_ylabel('DNL [LSB]')
    ax3[0].set_xlabel('ADC')
    ax3[0].set_ylabel('INL [LSB]')
    ax2[0].set_xlim(0,255)
    ax2[0].set_ylim(-0.5,0.5)
    ax3[0].set_xlim(0,255)
    ax3[0].set_ylim(-3,3)
    ax2[1].set_xlabel('DNL Extrema')
    ax3[1].set_xlabel('INL Extrema')
    ax2[1].set_ylabel('Channel Count')
    ax2[1].set_xlim(0.2,0.4)
    ax3[1].set_ylabel('Channel Count')
    ax3[1].set_xlim(2,4)
    ax2[1].legend()
    ax3[1].legend()

    plt.show()
        
        
    
def main(**kwargs):
    f = h5py.File('/home/russell/LArPix/v2_ASIC_paper/data/v2a/adc-linearity/f.h5','r');
    p = basic_parsing(f)
#    plot_adc_dist(p)
#    plot_timestamp_vs_adc(p)
    find_static_linearity(p)
    return



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
