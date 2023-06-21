import matplotlib.pyplot as plt
import numpy as np
import h5py

vref_dac=185 #255
vcm_dac=41 #50
#file_dir='/home/russell/LArPix/v2b-bare-die/cryo/adc-linearity/datalog_2022_01_24_22_16_27_PST.h5'
#file_dir='/home/russell/LArPix/v2b-bare-die/cryo/adc-linearity/datalog_2022_01_25_07_36_22_PST.h5'
#file_dir='/home/russell/LArPix/v2b-bare-die/cryo/adc-linearity/datalog_2022_01_18_10_00_11_PST.h5'
#file_dir='/home/russell/LArPix/v2b-bare-die/cryo/adc-linearity/datalog_2022_01_18_10_07_07_PST.h5'
file_dir='/home/russell/LArPix/v2b-bare-die/cryo/adc-linearity/datalog_2022_01_18_10_12_33_PST.h5'
#first_index=45000
#last_index=75000

first_index=4500
last_index=5800
first_adc=1
last_adc=254

def expected_dv_dt():
    return 1800./300. # mV/s


def report_lsb(verbose=False,vref=vref_dac, vcm=vcm_dac, gain=250., vdda=1770):
    vref_v = (vref/256.)*vdda
    vcm_v = (vcm/256.)*vdda
    if verbose: print('Vref: ',vref_v,'mV \t Vcm: ',vcm_v,'mV')
    return (vref-vcm)/256.


def plot_adc_vs_timestamp_single_channel(file_dir, channel):
    f = h5py.File(file_dir,'r')
    data_mask = f['packets'][:]['packet_type']==0
    valid_parity_mask = f['packets'][:]['valid_parity']==1
    channel_mask = f['packets'][:]['channel_id']==channel
    mask = np.logical_and(channel_mask, np.logical_and(data_mask, valid_parity_mask))
    data = f['packets'][mask] 
    adc = data['dataword']
    timestamp = data['timestamp']
    index = list(range(len(adc)))
    
    fig, ax = plt.subplots(2,2,figsize=(12,6))
    ax[0][0].plot(index, adc)
    ax[0][1].plot(index, timestamp)
    ax[1][0].plot(index[first_index:last_index], adc[first_index:last_index])
    ax[1][1].plot(index[first_index:last_index], timestamp[first_index:last_index])
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel('Index')
            ax[i][j].set_ylabel('ADC')
            ax[i][j].set_title('Channel '+str(channel))
            ax[i][j].grid(True)
            ax[i][j].axvline(x=4500,color='g',linestyle='--')
            ax[i][j].axvline(x=5800,color='r',linestyle='--')
    plt.show()
    #plt.savefig('channel'+str(channel)+'.png')

    
def plot_adc_single_channel(file_dir, channel):
    f = h5py.File(file_dir,'r')
    data_mask = f['packets'][:]['packet_type']==0
    valid_parity_mask = f['packets'][:]['valid_parity']==1
    channel_mask = f['packets'][:]['channel_id']==channel
    mask = np.logical_and(channel_mask, np.logical_and(data_mask, valid_parity_mask))
    adc = f['packets'][mask]['dataword'] 
    index = list(range(len(adc)))

    nbins = np.linspace(0,255,256)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(adc[first:last], bins=nbins); ax.set_xlabel('Dataword [ADC]'); ax.set_ylabel('Dataword Count'); ax.set_title('Channel '+str(channel))
    plt.savefig('adc_channel'+str(channel)+'.png')


def plot_static_linearity_single_channel(file_dir, channel, dv_dt, lsb, plot=True):
    f = h5py.File(file_dir,'r')
    data_mask = f['packets'][:]['packet_type']==0
    valid_parity_mask = f['packets'][:]['valid_parity']==1
    channel_mask = f['packets'][:]['channel_id']==channel
    mask = np.logical_and(channel_mask, np.logical_and(data_mask, valid_parity_mask))
    adc = f['packets'][mask]['dataword']
    values, bins = np.histogram(adc, bins=256, range=(0,255))

    avg=0
    for i in range(1,len(values)-1): avg+=values[i]
    #for i in range(1,len(values)): avg+=values[i]
    avg = avg / (len(values)-1)
    #avg = avg / (len(values)-1)
    dnl = [ (values[i]-avg)/avg for i in range(1,len(values)-1)]
                                    
    print('len values: ',len(values))
#    avg = np.sum(values)/len(values)
#    dnl = [ (v-avg)/avg for v in values]
    print('len dnl: ', len(dnl))
    print('range(len(dnl)): ', range(len(dnl)))
    inl = [ np.sum(dnl[0:i+1]) for i in range(len(dnl))]
    #inl = [ np.sum(dnl[0:i]) for i in range(len(dnl))]

    if plot:
        #nbins = np.linspace(first_adc,last_adc,last_adc)
        nbins = np.linspace(1,254,254)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(nbins,dnl,color='b'); ax.tick_params(axis='y', labelcolor='b')
        ax1 = ax.twinx()
        ax1.plot(nbins,inl,color='r'); ax1.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('Dataword [ADC]')
        ax.set_ylabel('DNL [LSB]',color='b'); ax1.set_ylabel('INL [LSB]',color='r')
        ax.set_title('Channel '+str(channel)); ax.grid(True)
        plt.savefig('linearity_channel'+str(channel)+'.png')    

    return dnl, inl

def plot_channel_linearity(linearity):
    fig, ax = plt.subplots(1,2,figsize=(16,6))
    x = [ key for key in linearity.keys() for i in linearity[key]['dnl']]
    y = [ i for key in linearity.keys() for i in linearity[key]['dnl']]
    ax[0].hist2d(x, y, bins=[64,200], range=[[0,63],[-2,2]])
    x = [ key for key in linearity.keys() for i in linearity[key]['inl']]
    y = [ i for key in linearity.keys() for i in linearity[key]['inl']]
    ax[1].hist2d(x, y, bins=[64,120], range=[[0,63],[-15,15]])
    for i in range(2): ax[i].set_xlabel('Channel ID')
    ax[0].set_ylabel('DNL [ADC]'); ax[1].set_ylabel('INL [ADC]')
    plt.savefig('summary.png')

def plot_dnl_hist(linearity):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist([i for key in linearity.keys() for i in linearity[key]['dnl']], np.linspace(-2,2,200))
    ax.set_xlabel('DNL [ADC]')
    plt.savefig('dnl.png')
    
def main():
#    dv_dt = expected_dv_dt()
#    lsb = report_lsb()
#    print('slope: ',dv_dt,' mV/s\t LSB: ',lsb,'mV/ADC')

#    linearity = dict()
    for i in range(3): #64):
        plot_adc_vs_timestamp_single_channel(file_dir, i)
#        dnl, inl = plot_static_linearity_single_channel(file_dir, i, dv_dt, lsb) #, False)
#        linearity[i] = dict( dnl = dnl, inl = inl )

    #plot_channel_linearity(linearity)
    #plot_dnl_hist(linearity)
    #for i in range(64):
    #    plot_adc_vs_timestamp_single_channel(file_dir, i)
    #plot_adc_single_channel(file_dir, i)





    
        
    
if __name__=='__main__':
    main()
