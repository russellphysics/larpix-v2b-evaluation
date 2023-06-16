import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import argparse
import common



def plot_analog_waveform():
    trace=input("File trace number: ")
    input_dir=input("Path to files: ")
    time, ch1, ch2, ch3 = common.parse_csv(input_dir, trace, 4, 21, 1e6, True)

    injection = np.min(ch2)
    min_injection_index = ch2.index(injection)
    injection_baseline = ch2[min_injection_index-1010:min_injection_index-10]

    response_baseline = ch1[min_injection_index-1010:min_injection_index-10]
    #injected_response = ch1[min_injection_index+50:min_injection_index+1050]
    #diff_response = np.average(response_baseline)-np.average(injected_response)
    peak_response = np.max(ch1)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(time, ch1, label='Analog Monitor')
    ax.plot(time, ch2, label='Charge Injection {:.1f} mV'.format(injection-np.average(injection_baseline)))
    ax.plot(time[min_injection_index-1010:min_injection_index-10], injection_baseline, label='Charge Injection Baseline {:.1f} mV'.format(np.average(injection_baseline)))    
    ax.plot(time, ch3, label='External Trigger')
    ax.plot(time[min_injection_index-1010:min_injection_index-10], response_baseline, label='Response Baseline {:.1f} mV'.format(np.average(response_baseline)))
#    ax.plot(time[min_injection_index+50:min_injection_index+1050],\
#            injected_response, label='Injection Response {:.1f} mV'.format(np.average(injected_response)))
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel(r'Amplitude [mV]')
    ax.set_title('{:.2f} mV Analog Response'.format(peak_response-np.average(response_baseline)))
    ax.legend()
    plt.show()

    

def create_analog_response_dictionary(file_path):
    analog_response=dict()
    for filename in glob.glob(file_path+'/24january2022/gain_analog_linearity/*/*.csv'):
        date = filename.split('/')[6]
        identifier = filename.split('/')[-1].split("ALL")[0].split("tek")[-1]
        gain = filename.split("/")[-2].split("_")[-1]
        injected_signal = filename.split("/")[-2].split("_")[-2]
        channel = filename.split("/")[-2].split("_")[-3]
        
        input_path=filename.split("/tek")[0]
        time, ch1, ch2, ch3 = common.parse_csv(input_path, identifier, \
                                               4, 21, 1e6, True)

        injection = np.min(ch2)
        min_inj_index = ch2.index(injection)
        injection_baseline = ch2[min_inj_index-1010:min_inj_index-10]

        response_baseline = ch1[min_inj_index-1010:min_inj_index-10]
        peak_response = np.max(ch1)

        analog_response[identifier]=dict(
            date=date,
            channel=channel.split("ch")[-1],
            gain=gain.split("uV")[0],
            resp_base_mu = np.mean(response_baseline),
            resp_base_sigma = np.std(response_baseline),
            resp = peak_response,
            inj_base_mu = np.mean(injection_baseline),
            inj_base_sigma = np.std(injection_baseline),
            inj = injection,
            inj_func_gen = injected_signal.split("mV")[0]
            )
        
    with open('v2b_gain_analog_linearity.json','w') as out:
        json.dump(analog_response, out, indent=4)

        
    
def main(single, file_path, **kwargs):
    if single: plot_analog_waveform()
    else: create_analog_response_dictionary(file_path)


    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', default=False, type=bool, \
                        help='''Plot single waveform''')
    parser.add_argument('--file_path', default='/home/russell/LArPix/v2_ASIC_paper/data')
    args = parser.parse_args()
    main(**vars(args))
