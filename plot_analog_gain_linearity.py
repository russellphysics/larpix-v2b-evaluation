import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
from scipy.optimize import curve_fit


def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def find_channels(d):
    channels=set()
    for key in d.keys(): channels.add(d[key]['channel'])
    return list(channels)



def plot_injected_baseline(d, channels):
    mu_bins=np.linspace(-20,0,41)
    std_bins=np.linspace(0,10,21)
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    for i in range(len(channels)):
        ax[0].hist([d[k]['inj_base_mu'] for k in d.keys() if d[k]['channel']==channels[i]], bins=mu_bins, histtype='step', label='Channel '+channels[i])
        ax[1].hist([d[k]['inj_base_sigma'] for k in d.keys() if d[k]['channel']==channels[i]], bins=std_bins, histtype='step', label='Channel '+channels[i])
        ax[2].scatter([d[k]['inj_base_mu'] for k in d.keys() if d[k]['channel']==channels[i]],\
                       [d[k]['inj_base_sigma'] for k in d.keys() if d[k]['channel']==channels[i]], label='Channel '+channels[i])

    for i in range(3):
        ax[i].legend()
        ax[i].grid(True)
    ax[0].set_xlabel('Injected Baseline Mean [mV]')
    ax[1].set_xlabel('Injected Baseline RMS [mV]')
    ax[2].set_xlabel('Injected Baseline Mean [mV]')
    ax[2].set_ylabel('Injected Baseline RMS [mV]')
    plt.show()



def plot_response_baseline(d, channels):
    mu_bins=np.linspace(500,540,81)
    std_bins=np.linspace(0,20,41)
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    for i in range(len(channels)):
        ax[0].hist([d[k]['resp_base_mu'] for k in d.keys() if d[k]['channel']==channels[i]], bins=mu_bins, histtype='step', label='Channel '+channels[i])
        ax[1].hist([d[k]['resp_base_sigma'] for k in d.keys() if d[k]['channel']==channels[i]], bins=std_bins, histtype='step', label='Channel '+channels[i])
        ax[2].scatter([d[k]['resp_base_mu'] for k in d.keys() if d[k]['channel']==channels[i]],\
                       [d[k]['resp_base_sigma'] for k in d.keys() if d[k]['channel']==channels[i]], label='Channel '+channels[i])

    for i in range(3):
        ax[i].legend()
        ax[i].grid(True)
    ax[0].set_xlabel('Response Baseline Mean [mV]')
    ax[1].set_xlabel('Response Baseline RMS [mV]')
    ax[2].set_xlabel('Response Baseline Mean [mV]')
    ax[2].set_ylabel('Response Baseline RMS [mV]')
    plt.show()
    


def plot_injected_versus_response(d, channels):
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    for i in range(len(channels)):
        ax[0].scatter([d[k]['inj']-d[k]['inj_base_mu'] for k in d.keys() \
                       if d[k]['channel']==channels[i] and d[k]['gain']=='2'],
                      [d[k]['resp']-d[k]['resp_base_mu'] for k in d.keys() \
                       if d[k]['channel']==channels[i] and d[k]['gain']=='2'],
                   label='Channel '+channels[i])
        ax[1].scatter([d[k]['inj']-d[k]['inj_base_mu'] for k in d.keys() \
                       if d[k]['channel']==channels[i] and d[k]['gain']=='4'],
                   [d[k]['resp']-d[k]['resp_base_mu'] for k in d.keys() \
                    if d[k]['channel']==channels[i] and d[k]['gain']=='4'], 
                   label='Channel '+channels[i])

    for i in range(2):
        ax[i].set_xlabel('Injected Signal [mV]')
        ax[i].set_ylabel('Channel Front-End Response [mV]')
        ax[i].legend()
        ax[i].grid(True)
    ax[0].set_title(r'2$\mu$V/e- Configured Gain')
    ax[1].set_title(r'4$\mu$V/e- Configured Gain')
    plt.show()


def find_injected_signal_per_channel(d, channels):
    output={}
    for chan in channels:
        for k in d.keys():
            temp=(d[k]['channel'],d[k]['gain'])
            if temp not in output.keys(): output[temp]=set()
            output[temp].add(int(d[k]['inj_func_gen']))

    for k in output.keys():
        output[k]=list(output[k])
        output[k].sort()
    return output

    
def fit_individual_input(d, channels):
    output={}
    ctr_dict={0:(0,0,np.linspace(20,50,21),np.linspace(60,90,21),20,50,21,60,90,21),
              1:(0,1,np.linspace(60,90,21),np.linspace(125,155,21),60,90,21,125,155,21),
              2:(0,2,np.linspace(270,300,21),np.linspace(255,285,21),270,300,21,255,285,21),
              3:(1,0,np.linspace(540,610,21),np.linspace(520,570,21),540,610,21,520,570,21),
              4:(1,1,np.linspace(1040,1130,21),np.linspace(980,1100,21),1040,1130,21,980,1100,21),
              5:(1,2,np.linspace(1080,1200,21),np.linspace(1040,1160,21),1080,1200,21,1040,1160,21)}
    inj_dict={0:(np.linspace(-30,-20,21),np.linspace(-30,-20,21),-30,-20,21,-30,-20,21),
              1:(np.linspace(-55,-45,21),np.linspace(-55,-45,21),-55,-45,21,-55,-45,21),
              2:(np.linspace(-220,-200,21),np.linspace(-110,-90,21),-220,-200,21,-110,-90,21),
              3:(np.linspace(-440,-400,21),np.linspace(-230,-190,40),-440,-400,21,-230,-190,21),
              4:(np.linspace(-840,-800,21),np.linspace(-430,-400,21),-840,-800,21,-430,-400,21),
              5:(np.linspace(-950,-900,21),np.linspace(-480,-450,21),-950,-900,21,-480,-450,21)}
    ch_gain = find_injected_signal_per_channel(d, channels)
    fig0, ax0 = plt.subplots(2,3,figsize=(18,12))
    fig1, ax1 = plt.subplots(2,3,figsize=(18,12))
    fig2, ax2 = plt.subplots(2,3,figsize=(18,12))
    fig3, ax3 = plt.subplots(2,3,figsize=(18,12))

    for cg in ch_gain.keys():
        ctr=0
        for inj in ch_gain[cg]:
            x=[d[k]['resp']-d[k]['resp_base_mu'] for k in d.keys() \
               if d[k]['channel']==cg[0] and d[k]['gain']==cg[1] \
               and d[k]['inj_func_gen']==str(inj)]
            x0=[d[k]['inj']-d[k]['inj_base_mu'] for k in d.keys() \
                if d[k]['channel']==cg[0] and d[k]['gain']==cg[1] \
                and d[k]['inj_func_gen']==str(inj)]
            if cg[1]=='2':
                hist, bin_edges = np.histogram(x, bins=21, range=(ctr_dict[ctr][4], ctr_dict[ctr][5]))
                params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                                        p0=[max(hist), np.mean(ctr_dict[ctr][2]), np.std(ctr_dict[ctr][2])])
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].plot(bin_edges[:-1], \
                                                             gauss(bin_edges[:-1], *params), \
                                                             '-', label=r'Fit channel {} $\mu$={:.1f}$\pm${:.1f} $\sigma$={:.1f}$\pm${:.1f}'.format(cg[0], params[1], np.sqrt(cov[1][1]),params[2],np.sqrt(cov[2][2])))
                resp_mu=params[1]; resp_mu_error=np.sqrt(cov[1][1])
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].hist(x,
                                                             bins=ctr_dict[ctr][2],
                                                             histtype='step', label='Channel '+cg[0])
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_title(str(inj)+' mV Injected Pulse')
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_ylabel('Analog Traces [count]')
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_xlabel('Front End Response [mV]')
                ax0[ctr_dict[ctr][0]][ctr_dict[ctr][1]].legend()
                fig0.suptitle(r'2 $\mu$V/e$^-$ Configured Gain'+'\n'+'Response Signal')

                hist, bin_edges = np.histogram(x0, bins=21, range=(inj_dict[ctr][2], inj_dict[ctr][3]))
                params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                                        p0=[max(hist), np.mean(inj_dict[ctr][0]), np.std(inj_dict[ctr][0])])
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].plot(bin_edges[:-1], \
                                                             gauss(bin_edges[:-1], *params), \
                                                             '-', label=r'Fit channel {} $\mu$={:.1f}$\pm${:.1f} $\sigma$={:.1f}$\pm${:.1f}'.format(cg[0], params[1], np.sqrt(cov[1][1]),params[2],np.sqrt(cov[2][2])))
                inj_mu=params[1]; inj_mu_error=np.sqrt(cov[1][1])
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].hist(x0,
                                                             bins=inj_dict[ctr][0],
                                                             histtype='step', label='Channel '+cg[0])
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_title(str(inj)+' mV Injected Pulse')
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_ylabel('Analog Traces [count]')
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_xlabel('Front End Response [mV]')
                ax2[ctr_dict[ctr][0]][ctr_dict[ctr][1]].legend()
                fig2.suptitle(r'2 $\mu$V/e$^-$ Configured Gain'+'\n'+'Injected Signal')

                # key: (channel, gain, injected function generator)
                # value: (mu, mu_error, input, input_error)
                if resp_mu_error<resp_mu:
                    if inj_mu_error<abs(inj_mu):
                        output[(cg[0], cg[1], inj)]=(resp_mu, resp_mu_error, inj_mu, inj_mu_error) 
                
            if cg[1]=='4':
                hist, bin_edges = np.histogram(x, bins=21, range=(ctr_dict[ctr][7], ctr_dict[ctr][8]))
                params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                                        p0=[max(hist), np.mean(ctr_dict[ctr][3]), np.std(ctr_dict[ctr][3])])
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].plot(bin_edges[:-1], \
                                                             gauss(bin_edges[:-1], *params), \
                                                             '-', label=r'Fit channel {} $\mu$={:.1f}$\pm${:.1f} $\sigma$={:.1f}$\pm${:.1f}'.format(cg[0], params[1], np.sqrt(cov[1][1]),params[2],np.sqrt(cov[2][2])))
                resp_mu=params[1]; resp_mu_error=np.sqrt(cov[1][1])
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].hist(x,
                                                             bins=ctr_dict[ctr][3],
                                                             histtype='step', label='Channel '+cg[0])
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_title(str(inj)+' mV Injected Pulse')
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_ylabel('Analog Traces [count]')
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_ylabel('Front End Response [mV]')
                ax1[ctr_dict[ctr][0]][ctr_dict[ctr][1]].legend()
                fig1.suptitle(r'4 $\mu$V/e$^-$ Configured Gain'+'\n'+'Response Signal')

                hist, bin_edges = np.histogram(x0, bins=21, range=(inj_dict[ctr][5], inj_dict[ctr][6]))
                params, cov = curve_fit(gauss, bin_edges[:-1], hist, \
                                        p0=[max(hist), np.mean(inj_dict[ctr][1]), np.std(inj_dict[ctr][1])])
                
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].plot(bin_edges[:-1], \
                                                             gauss(bin_edges[:-1], *params), \
                                                             '-', label=r'Fit channel {} $\mu$={:.1f}$\pm${:.1f} $\sigma$={:.1f}$\pm${:.1f}'.format(cg[0], params[1], np.sqrt(cov[1][1]),params[2],np.sqrt(cov[2][2])))
                inj_mu=params[1]; inj_mu_error=np.sqrt(cov[1][1])                
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].hist(x0,
                                                             bins=inj_dict[ctr][1],
                                                             histtype='step', label='Channel '+cg[0])
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_title(str(inj)+' mV Injected Pulse')
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_ylabel('Analog Traces [count]')
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].set_xlabel('Front End Response [mV]')
                ax3[ctr_dict[ctr][0]][ctr_dict[ctr][1]].legend()
                fig3.suptitle(r'4 $\mu$V/e$^-$ Configured Gain'+'\n'+'Injected Signal')
                # key: (channel, gain, injected function generator)
                # value: (mu, mu_error, input, input_error)
                if resp_mu_error<resp_mu:
                    if inj_mu_error<abs(inj_mu):
                        output[(cg[0], cg[1], inj)]=(resp_mu, resp_mu_error, inj_mu, inj_mu_error) 
            ctr+=1
#    plt.show()
    return output


def plot_linear_fit(fs):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    #fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    #fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    channels=set([k[0] for k in fs.keys()])
    gains=set([k[1] for k in fs.keys()])
    for ch in channels:
        for g in gains:
            x = [fs[k][2] for k in fs.keys() if k[0]==ch and k[1]==g]
            x_electrons = [fs[k][2] for k in fs.keys() if k[0]==ch and k[1]==g]
            x_err = [fs[k][3] for k in fs.keys() if k[0]==ch and k[1]==g]
            y = [fs[k][0] for k in fs.keys() if k[0]==ch and k[1]==g]
            y_err = [fs[k][1] for k in fs.keys() if k[0]==ch and k[1]==g]
            if g=='2':
                ax[0].errorbar(x, y, yerr=y_err, xerr=x_err, linestyle="", label='Channel '+ch)
                coef=np.polyfit(x,y,1)
                poly1d_fn = np.poly1d(coef)
                ax[0].plot(x, poly1d_fn(x), '--', label='Channel {} slope={:.2f} y-intercept={:.2f}'.format(ch, coef[0], coef[1]))
                #ax1[0].plot(range(len(x_err)), x_err, label='Channel '+ch)
                #ax2[0].plot(range(len(y_err)), y_err, label='Channel '+ch)
            if g=='4':
                ax[1].errorbar(x, y, yerr=y_err, xerr=x_err, linestyle="", label='Channel '+ch)
                coef=np.polyfit(x,y,1)
                poly1d_fn = np.poly1d(coef)
                ax[1].plot(x, poly1d_fn(x), '--', label='Channel {} slope={:.2f} y-intercept={:.2f}'.format(ch, coef[0], coef[1]))
                #ax1[1].plot(range(len(x_err)), x_err, label='Channel '+ch)
                #ax2[1].plot(range(len(y_err)), y_err, label='Channel '+ch)                            
    for i in range(2):
        if i==0:
            ax[0].set_title(r'2 $\mu$V/e$^-$ Configured Gain')
            ax3[0].set_title(r'2 $\mu$V/e$^-$ Configured Gain')
        if i==1:
            ax[1].set_title(r'4 $\mu$V/e$^-$ configured Gain')
            ax3[1].set_title(r'4 $\mu$V/e$^-$ configured Gain')
        ax[i].set_xlabel('Input Signal [mV]')
        ax[i].set_ylabel('Analog Response [mV]')
        ax[i].grid(True)
        ax[i].legend()
        ax3[i].set_xlabel(r'External Test Pulse [ke$^-$]')
        ax3[i].set_ylabel('Channel Analog Response [mV]')
        ax3[i].grid(True)
#        ax3[i].legend()
    plt.show()


    
def plot_response(d):
    channels = find_channels(d)

#    plot_injected_baseline(d, channels)
#    plot_response_baseline(d, channels)
    fit_summary = fit_individual_input(d, channels)
    plot_linear_fit(fit_summary)
    #plot_injected_versus_response(d, channels)

        

    
def main(input_json, **kwargs):
    with open(input_json) as fin:
        ana_gain = json.load(fin)

    plot_response(ana_gain)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', \
                        default='v2b_gain_analog_linearity.json',\
                       type=str, help='''Input JSON file''')
    args = parser.parse_args()
    main(**vars(args))
                       
