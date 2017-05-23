from scipy.signal import butter, filtfilt, decimate
import numpy as np

def filt(sig, low, high, fs, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)


def extract_freqs(sig, bands, fs, normalize=True, filter_order=3, dec=1):
    if dec > 1:
        fs /= dec
        sig = decimate(sig, dec, zero_phase=True)
    filt_sigs = []
    norm_consts = []
    for band in bands:
        filt_sig = filt(sig, band[0], band[1], fs, filter_order)
        filt_sigs.append(filt_sig)
        norm_consts.append(np.max(np.abs(filt_sig)))
        if normalize:
            filt_sig /= np.max(np.abs(filt_sig))
    return filt_sigs, norm_consts
