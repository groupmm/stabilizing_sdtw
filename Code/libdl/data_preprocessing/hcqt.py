import os
import numpy as np, os, scipy, scipy.spatial, matplotlib.pyplot as plt, IPython.display as ipd
import librosa
import pandas as pd, pickle, re
from numba import jit



@jit(nopython=True)
def compute_hopsize_cqt(fs_cqt_target, fs=22050, num_octaves=7):
    """ Computes the necessary CQT hopsize to approximate a desired feature rate fs_cqt_target.

    Args:
        fs_cqt_target:   desired frame rate in Hz
        fs:              audio sampling rate
        num_octaves:     number of octaves for the CQT

    Returns:
        hopsize_cqt:     CQT hopsize in samples
        fs_cqt:          resulting CQT frame rate in Hz

    """

    factor = 2**(num_octaves-1)
    hopsize_target = fs / fs_cqt_target
    n = np.round(hopsize_target/factor)
    hopsize_cqt = int(np.max(np.array([1, factor*n])))
    fs_cqt = fs/hopsize_cqt

    return hopsize_cqt, fs_cqt



def compute_hcqt(f_audio, fs=22050, fmin=librosa.note_to_hz('C1'), fs_hcqt_target=91, bins_per_octave=60,
                 num_octaves=6, num_harmonics=5, num_subharmonics=1, center_bins=True):
    """ Computes a standard HCQT with one inidividual CQT for each (sub)harmonic.

    Args (defaults: HCQT by Bittner et al. ISMIR 2017):
        f_audio:           desired feature rate in Hz
        fs:                sampling rate of input audio
        fmin:              base frequency of first harmonic (fundamental)
        fs_hcqt_target:    desired frame rate of HCQT in Hz
        num_octaves:       number of octaves for each CQT
        bins_per_octave:   number of bins per octave
        num_harmonics:     number of harmonics
        num_subharmonics:  number of subharmonics
        center_bins:       flag to center bins to MIDI notes

    Returns:
        f_hcqt:            HCQT tensor, dimensions "#pitch_bins * #time_frames * #(sub)harmonics"
        fs_hcqt:           resulting HCQT frame rate in Hz
        hopsize_hcqt:      resulting HCQT hopsize in samples
    """
    hopsize_cqt, fs_cqt = compute_hopsize_cqt(fs_hcqt_target, fs=fs, num_octaves=num_octaves)
    fs_hcqt = fs/hopsize_cqt
    n_bins = num_octaves*bins_per_octave
    assert np.mod(bins_per_octave, 12)==0, 'Error: bins_per_octave no multiple of 12'
    bins_per_semitone = int(bins_per_octave/12)

    if center_bins:
        fmin = fmin / 2**((bins_per_semitone-1)/(2*bins_per_octave))

    tuning_est = librosa.estimate_tuning(y=f_audio, bins_per_octave=bins_per_octave)
    fmin_tuned = fmin * 2**(tuning_est / bins_per_octave)

    f_cqt = librosa.cqt(f_audio, sr=fs, hop_length=hopsize_cqt, fmin=fmin_tuned, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, tuning=0.0)
    n_frames = f_cqt.shape[1]

    f_hcqt = np.zeros((n_bins, n_frames, num_harmonics+num_subharmonics))
    f_hcqt[:, :, num_subharmonics] = np.abs(f_cqt)

    for n_ha in range(2, num_harmonics+1):
        fmin_ha = n_ha*fmin_tuned
        f_cqt_ha = librosa.cqt(f_audio, sr=fs, hop_length=hopsize_cqt, fmin=fmin_ha, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, tuning=0.0)
        f_hcqt[:, :, num_subharmonics+n_ha-1] = np.abs(f_cqt_ha)

    for n_hs in range(1, num_subharmonics+1):
        fmin_hs = fmin_tuned/(n_hs+1)
        f_cqt_hs = librosa.cqt(f_audio, sr=fs, hop_length=hopsize_cqt, fmin=fmin_hs, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, tuning=0.0)
        f_hcqt[:, :, num_subharmonics-n_hs] = np.abs(f_cqt_hs)

    return f_hcqt, fs_hcqt, hopsize_cqt



def compute_efficient_hcqt(f_audio, fs=22050, fmin=librosa.note_to_hz('C1'), fs_hcqt_target=91, bins_per_octave=60,
                           num_octaves=6, num_harmonics=5, num_subharmonics=1, center_bins=True):
    """ Computes an HCQT in an efficient way using the same CQT for multiple-of-two harmonics.

    Args (defaults: HCQT by Bittner et al. ISMIR 2017):
        f_audio:           desired feature rate in Hz
        fs:                sampling rate of input audio
        fmin:              base frequency of first harmonic (fundamental)
        fs_hcqt_target:    desired frame rate of HCQT in Hz
        num_octaves:       number of octaves for each CQT
        bins_per_octave:   number of bins per octave
        num_harmonics:     number of harmonics
        num_subharmonics:  number of subharmonics
        center_bins:       flag to center bins to MIDI notes

    Returns:
        f_hcqt:            HCQT tensor, dimensions "#pitch_bins * #time_frames * #(sub)harmonics"
        fs_hcqt:           resulting HCQT frame rate in Hz
        hopsize_hcqt:      resulting HCQT hopsize in samples
    """

    eps = np.finfo(float).eps

    # Compute effective number of octaves necessary for efficient HCQT
    num_octaves_eff = num_octaves + np.ceil(np.log2((num_subharmonics+1))+np.log2((num_harmonics))).astype(int)
    hopsize_cqt, fs_cqt = compute_hopsize_cqt(fs_hcqt_target, fs=fs, num_octaves=num_octaves_eff)
    fs_hcqt = fs/hopsize_cqt
    assert np.mod(bins_per_octave, 12)==0, 'Error: bins_per_octave no multiple of 12'
    bins_per_semitone = int(bins_per_octave/12)

    if center_bins:
        fmin = fmin / 2**((bins_per_semitone-1)/(2*bins_per_octave))

    tuning_est = librosa.estimate_tuning(y=f_audio, bins_per_octave=bins_per_octave)
    fmin_tuned = fmin * 2**(tuning_est / bins_per_octave)

    n_frames = np.floor(f_audio.shape[0]/hopsize_cqt).astype(int)+1
    n_bins = bins_per_octave*num_octaves
    f_hcqt = np.zeros((n_bins, n_frames, num_harmonics+num_subharmonics))

    list_harmonics = [1/(n_sh+1) for n_sh in range(num_subharmonics, 0, -1) ] + [n_ha for n_ha in range(1, num_harmonics+1)]
    base_harmonics = np.zeros((len(list_harmonics)))
    computed_harmonics = np.zeros((len(list_harmonics)))

    base_harmonics[0] = 1/(num_subharmonics+1)
    computed_harmonics[0] = 1

    for n_h in range(1, len(list_harmonics)):
        harmonic = list_harmonics[n_h]
        n_base = 0
        while computed_harmonics[n_h]<eps:
            base = base_harmonics[n_base]
            if base==0:
                base_harmonics[n_h] = list_harmonics[n_h]
                computed_harmonics[n_h]=1
            elif np.mod(np.log2(harmonic/base), 1)==0:
                base_harmonics[n_h] = base
                computed_harmonics[n_h]=1
            else:
                n_base += 1

    for base_h in np.unique(base_harmonics):
        fmin_h = fmin_tuned*base_h
        all_harmonics = np.where(base_harmonics==base_h)[0]
        max_harmonic = np.max(all_harmonics)
        num_add_octaves = int(np.ceil(np.log2(list_harmonics[max_harmonic]/base_h)))
        num_octaves_curr = num_octaves + num_add_octaves
        n_bins = num_octaves_curr*bins_per_octave
        f_cqt_h = librosa.cqt(f_audio, sr=fs, hop_length=hopsize_cqt, fmin=fmin_h, n_bins=n_bins,
                              bins_per_octave=bins_per_octave, tuning=0.0)
        for harmonic in np.array(list_harmonics)[base_harmonics==base_h]:
            factor = np.log2(harmonic/base_h).astype(int)
            harm_index = int(np.where(np.array(list_harmonics)==harmonic)[0])
            f_hcqt[:, :, harm_index] = np.abs(f_cqt_h[factor*bins_per_octave:(factor+num_octaves)*bins_per_octave,:])

    return f_hcqt, fs_hcqt, hopsize_cqt



