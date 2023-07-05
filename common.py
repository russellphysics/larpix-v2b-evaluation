import numpy as np
import csv
import glob
import matplotlib.pyplot as plt

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def fig_ax_sixty_four_channel(width, length):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, \
                           figsize=(width,length))
    d={0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4), 5:(0,5), 6:(0,6), 7:(0,7),
       8:(1,0), 9:(1,1), 10:(1,2), 11:(1,3), 12:(1,4), 13:(1,5), 14:(1,6), 15:(1,7),
       16:(2,0), 17:(2,1), 18:(2,2), 19:(2,3), 20:(2,4), 21:(2,5), 22:(2,6), 23:(2,7),
       24:(3,0), 25:(3,1), 26:(3,2), 27:(3,3), 28:(3,4), 29:(3,5), 30:(3,6), 31:(3,7),
       32:(4,0), 33:(4,1), 34:(4,2), 35:(4,3), 36:(4,4), 37:(4,5), 38:(4,6), 39:(4,7),
       40:(5,0), 41:(5,1), 42:(5,2), 43:(5,3), 44:(5,4), 45:(5,5), 46:(5,6), 47:(5,7),
       48:(6,0), 49:(6,1), 50:(6,2), 51:(6,3), 52:(6,4), 53:(6,5), 54:(6,6), 55:(6,7),
       56:(7,0), 57:(7,1), 58:(7,2), 59:(7,3), 60:(7,4), 61:(7,5), 62:(7,6), 63:(7,7)}
    return fig, ax, d


def unique_channel_id(io_group, io_channel, chip_id, channel_id):
    return channel_id + 100*(chip_id + 1000*(io_channel + 1000*(io_group)))


def unique_2_channel_id(unique): return unique % 100


def unique_2_chip_id(unique): return (unique//100) % 1000


def unique_2_io_channel(unique): return (unique//(100*1000)) % 1000


def unique_2_io_group(unique): return (unique//(100*1000*1000)) % 1000



def basic_parsing(f):
    parity_mask=f['packets'][:]['valid_parity']==1
    data_mask=f['packets'][:]['packet_type']==0
    mask = np.logical_and(parity_mask, data_mask)
    return f['packets'][mask]
    



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


def parse_csv(input_path, identifier, n_columns, n_lines_ignore, timescale, verbose):
    if n_columns<=1 or n_columns>=6:
        print('Incompatible columns specified')
        return
    
    locate = str(input_path)+'/*'+identifier+'*.csv'
    time, ch1, ch2, ch3, ch4 = [[] for i in range(5)]
    for filename in glob.glob(locate):
        if verbose: print('processing ',filename)
        with open(filename,'r') as fin:
            info = csv.reader(fin)
            row_counter=0
            for row in info:
                row_counter+=1
                if row_counter<=n_lines_ignore: continue
                if len(row)!=n_columns: continue

                time.append(float(row[0])*timescale)
                ch1.append(float(row[1])*1e3)
                if n_columns>2: ch2.append(float(row[2])*1e3)
                if n_columns>3: ch3.append(float(row[3])*1e3)
                if n_columns>4: ch4.append(float(row[4])*1e3)
    if n_columns==2: return time, ch1
    if n_columns==3: return time, ch1, ch2
    if n_columns==4: return time, ch1, ch2, ch3
    if n_columns==5: return time, ch1, ch2, ch3, ch4
                
    
