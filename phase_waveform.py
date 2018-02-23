import pandas as pd
import numpy as np
import struct
import codecs
from scipy import signal
import h5py


class PhaseWaveform(object):
    """
    PhaseWaveform
    """
    phases = ['regP', 'regS', 'tele', 'N']
    phase_index = {phase: index for index, phase in enumerate(phases)}
    channels = ["BHE", "BHZ", "BHN"]
    channel_index = {channel: index for index, channel in enumerate(channels)}

    def __init__(self, filename_features, filename_waveforms, random_state=1):
        """
        :param filename:
        :param random_state:
        """
        self.dff = pd.read_csv(filepath_or_buffer=filename_features)
        self.dfw = pd.read_csv(filepath_or_buffer=filename_waveforms)
        self.random_state = random_state

    def get_waveforms(self, arid):
        """
        Returns waveforms for all three channels for a given arrival
        :param arid:
        :param chan:
        :return:
        """
        rows = (self.dfw.ARID == arid)
        data = self.dfw[rows].values

        print('Arid:{}, fetched {} waveforms'.format(arid, len(data)))

        waveforms = [None, None, None]

        for dat in data:
            nsamp = int(dat[7])
            chan = dat[3]
            if nsamp == 0:
                continue
            try:
                waveform = np.array(struct.unpack('%sf' % nsamp, codecs.decode(dat[9], 'hex_codec')))
            except struct.error as err:
                print(err)
                continue
            waveforms[self.channel_index[chan]] = waveform

        return waveforms

    def get_wavelets(self, arid, chan=None):
        waveforms = self.get_waveforms(arid)
        wavelets = [None, None, None]
        for i, waveform in enumerate(waveforms):
            if waveform is None:
                continue
            widths = np.arange(1, 41)
            wavelet = signal.cwt(waveform, signal.ricker, widths)
            wavelets[i] = wavelet

        return wavelets

    def save_wavelets(self, filename):
        with h5py.File(filename, "w") as f:
            station = f.create_group("station")
            # urz = station.create_group("URZ")
            # lpaz = station.create_group("LPAZ")

            phase_counter = {}
            counter = 0
            # for arid in sorted(list(set(self.dfw.ARID))):
            arids = list(set(self.dfw.ARID))
            for arid in arids:
                print("arid:{}, counter:{}".format(arid, counter))
                dff_current = self.dff[(self.dff.ARID == arid)]
                dfw_current = self.dfw[(self.dfw.ARID == arid) & (self.dfw.SAMPRATE>0)]
                if len(dff_current) > 0 and len(dfw_current) >0:
                    phase = dff_current["CLASS_PHASE"].values[0]
                    station = dff_current["STA"].values[0]
                    if phase not in phase_counter:
                        phase_counter[phase] = 1
                    else:
                        phase_counter[phase] += 1
                    # print("{}:{}".format(arid, phase))
                    wavelets = self.get_wavelets(arid)
                    if any(wavelet is None for wavelet in wavelets):
                        continue
                    group_arid = f.create_group("/station/{}/{}".format(station, arid))
                    group_arid.create_dataset("phase", data=phase)
                    group_arid.create_dataset("wavelet", data=wavelets)
                    counter += 1


if __name__ == "__main__":
    FEATURES_TINY = "data/phase/ml_features_tiny.csv"
    FEATURES = "data/phase/ml_features.csv"
    WAVEFORMS_TINY = "data/phase/ml_waveforms_tiny.csv"
    WAVEFORMS = "data/phase/ml_waveforms.csv"
    WAVELETS = "data/phase/wavelets.hdf5"

    pw = PhaseWaveform(filename_features=FEATURES, filename_waveforms=WAVEFORMS)
    pw.save_wavelets(WAVELETS)