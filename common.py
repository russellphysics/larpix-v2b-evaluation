import numpy as np
import csv
import glob


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
                
    
