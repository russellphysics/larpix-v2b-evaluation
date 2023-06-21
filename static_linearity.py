import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

def adc_to_e(x, vref_dac=185., vcm_dac=41., vdda=1780., gain=240.):
    vref = (vref_dac/256.)*vdda
    vcm = (vcm_dac/256.)*vdda
    lsb = (vref-vcm)/256.
    e_enc = [float(i)*lsb*gain for i in x]
    return e_enc
    #return ["%.1f" % j for j in e_enc]


maxADC=220


fA = h5py.File('f.h5','r'); first_ctrA=2e6; last_ctrA=3.25e6
test_channels = [15,30,45,60,7,23,39,55,11,27,43,59]

data_adcA, data_tsA, data_indexA = [{} for i in range(3)]
for i in test_channels: data_adcA[i] = []; data_tsA[i]=[]; data_indexA[i]=[]

parity_mask = fA['packets'][:]['valid_parity']==1
type_mask = fA['packets'][:]['packet_type']==0
mask = np.logical_and(parity_mask,type_mask)
packets = fA['packets'][mask]
ctr=0
for i in range(len(packets)):
    if ctr>=first_ctrA and ctr<=last_ctrA:
        channel = packets[i]['channel_id']
        ts = packets[i]['timestamp']
        adc = packets[i]['dataword']
        if channel not in test_channels: continue
        data_tsA[int(channel)].append(int(ts))
        data_adcA[int(channel)].append(int(adc))
        data_indexA[int(channel)].append(ctr)
    ctr+=1

nbins=np.linspace(0,256,257)
fig2, ax2  = plt.subplots(1,3,figsize=(24,6))
chan_ctr=0

for i in range(3):
    if i>0: break
    for j in range(4):
        if j>0: break
        histA, bin_edgesA = np.histogram(data_adcA[test_channels[chan_ctr]],bins=nbins)
        meanDict, runningSum = [{} for k in range(2)]
        dnl_val, dnl_adc, inl_val = [{} for k in range(3)]
        for k in range(1,255):
            meanDict[k]=0; runningSum[k]=0
            dnl_val[k]=[]; dnl_adc[k]=[]; inl_val[k]=[]
        
        for k in range(len(histA)):
            if k==0 or k==255: continue
            for l in range(1,255):
                if l>=k: meanDict[l] += histA[k]

        for key in meanDict.keys(): meanDict[key] = meanDict[key]/key

        for k in range(len(histA)):
            if k==0 or k==255: continue
            for l in range(1,255):
                if l>=k:
                    value = (histA[k]-meanDict[l]) / meanDict[l] 
                    dnl_val[l].append( value )
                    runningSum[l]+=value
                    inl_val[l].append( runningSum[l] )
                    dnl_adc[l].append(k)

        ax2[0].grid(True)
        ax2[0].plot(dnl_adc[254],dnl_val[254],'-',color='k')
        ax2[0].set_xlabel('ADC',fontsize=14)
        ax2[0].set_ylabel('Differential Nonlinearity [LSB]',fontsize=14)
        #ax2[0].set_title('Channel '+str(test_channels[chan_ctr]))
                    
        ax2[1].grid(True)
        ax2[1].plot(dnl_adc[254],inl_val[254],'-',color='k')
        ax2[1].set_xlabel('ADC',fontsize=14)
        ax2[1].set_ylabel('Integral Nonlinearity [LSB]',fontsize=14)

        ax2[2].grid(True)
        e_from_adc = adc_to_e( dnl_adc[254] )
        res = [ ((np.abs(z)+0.94375)/254.)*100  for z in inl_val[254]]
        #res = ["%.2f" % r for r in res]
        ke_from_adc = [i/1000. for i in e_from_adc]
        ax2[2].plot( ke_from_adc, res, '-', color='k')
        ax2[2].set_xlabel('Charge [ke-]',fontsize=14)
        ax2[2].set_ylabel('Pixel Charge Resolution [%]',fontsize=14)
        #ax2[2].set_params(nbins=6, axis='x')
        max_yticks=15
        yloc=plt.MaxNLocator(max_yticks)
        ax2[2].yaxis.set_major_locator(yloc)
        max_xticks=6
        xloc=plt.MaxNLocator(max_xticks)
        #ax2[2].set_ylim(0,2.)
        ax2[2].xaxis.set_major_locator(xloc)
        ax2[2].axvline(x=180,label='180 ke-',linestyle='dashed',color='red')
        ax2[2].legend(loc='upper left')
        #ax2[2].ticklabel_format(axis='x',style='scientific')

        #ax2[1].set_title('Channel '+str(test_channels[chan_ctr]))
        #ax2[1].legend()
        #axA = ax2[i][j].twinx()
        #for k in range(10,230,10):
        #    adcs = np.array(dnl_adc[20])
        #    inls = np.array(inl_val[20])
        #    axA.plot(adcs,inls,'-',color='lightcoral')            

        #for k in range(10,230,10):
        #    axA.plot(dnl_adc[k],inl_val[k],'-',color='lightcoral')
        #axA.plot(dnl_adcAA,inl_valAA,'-',color='r')
        #axA.set_ylabel('INL [LSB]',color='r')

        chan_ctr+=1
plt.tight_layout()
plt.show()
