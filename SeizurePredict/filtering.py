from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.base import TransformerMixin, BaseEstimator

## -- PREPROCESSING FUNCTIONS --

# Highpass filter
def highpass_filter(signals, sampling_rate, hp_frequency = 0.1):
    sos = butter(N = 3, Wn = hp_frequency, btype="highpass",fs=sampling_rate, output="sos")
    filter_hp = sosfiltfilt(sos, signals)
    return filter_hp

# Powerline filter
def notch_filter(signals, sampling_rate, notch_frequency = 50, quality_factor = 30):
    w0 = notch_frequency/(sampling_rate/2)
    b_notch, a_notch = iirnotch(w0, quality_factor)
    filter_notch = filtfilt(b_notch, a_notch, signals, axis = -1)
    return filter_notch

# Create our own scaler
class CustomTranformer(TransformerMixin, BaseEstimator):
    # BaseEstimator generates the get_params() and set_params() methods that all Pipelines require
    # TransformerMixin creates the fit_transform() method from fit() and transform()

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.means = X.mean()
        return self

    def transform(self, X, y=None):
        norm_features = X - self.means
        return norm_features

# Combination of all filters and Scaler
def filter_signals(signals, sampling_rate, scaler, hp_frequency = 0.1, notch_frequency = 50, quality_factor = 30):
    filter_hp = highpass_filter(signals, sampling_rate, hp_frequency)
    filter_notch = notch_filter(filter_hp, sampling_rate, notch_frequency, quality_factor)
    final_signal = scaler.fit_transform(filter_notch)
    return final_signal
