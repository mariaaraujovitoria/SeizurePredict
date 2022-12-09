import pyedflib

def read_reduce_save_edf():
    path_raw_data = '/home/lee/code/mariaaraujovitoria/SeizurePredict/raw_data/test_eeg5.edf'
    path_new_file = '/home/lee/code/mariaaraujovitoria/SeizurePredict/raw_data/test2_eeg5.edf'

    signals, signal_headers, header = pyedflib.highlevel.read_edf(path_raw_data)

    signals2 = signals[:,:500000]

    res = pyedflib.highlevel.write_edf(path_new_file, signals=signals2, signal_headers=signal_headers, header=header)
    print(res)

if __name__ == "__main__":
    read_reduce_save_edf()
