import argparse
import pandas as pd
import numpy as np
import struct
import codecs
from scipy import signal
import h5py
import math

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

    def get_wavelets(self, arid, logarithmic=True, log_after=True):
        waveforms = self.get_waveforms(arid)
        wavelets = [None, None, None]
        for i, waveform in enumerate(waveforms):
            if waveform is None:
                continue
            if logarithmic and not log_after:
                for j, value in enumerate(waveform):
                    if value >= 0:
                        value = math.log10(value+1)
                    else:
                        value = - math.log10(abs(value)+1)
                    waveform[j] = value
            widths = np.arange(1, 41)
            wavelet = signal.cwt(waveform, signal.ricker, widths)
            if logarithmic and log_after:
                wavelet_row, wavelet_columns = wavelet.shape
                for k in range(wavelet_row):
                    for l in range(wavelet_columns):
                        if wavelet[k][l] >= 0:
                            wavelet[k][l] = math.log10(wavelet[k][l]+1)
                        else:
                            wavelet[k][l] = -math.log10(abs(wavelet[k][l])+1)
            wavelets[i] = wavelet

        return wavelets

    def save_wavelets(self, filename, logarithmic=True, log_after=True):
        with h5py.File(filename, "w") as f:
            station = f.create_group("station")
            # urz = station.create_group("URZ")
            # lpaz = station.create_group("LPAZ")

            phase_counter = {}
            counter = 0
            # for arid in sorted(list(set(self.dfw.ARID))):
            arids = list(set(self.dfw.ARID))
            for arid in arids:
                # if counter > 1000:
                #    break
                print("arid:{}, counter:{}".format(arid, counter))
                dff_current = self.dff[(self.dff.ARID == arid)]
                dfw_current = self.dfw[(self.dfw.ARID == arid) & (self.dfw.SAMPRATE > 0)]
                if len(dff_current) > 0 and len(dfw_current) > 0:
                    phase = dff_current["CLASS_PHASE"].values[0]
                    station = dff_current["STA"].values[0]
                    source = dff_current["SOURCE"].values[0]
                    if phase not in phase_counter:
                        phase_counter[phase] = 1
                    else:
                        phase_counter[phase] += 1
                    # print("{}:{}".format(arid, phase))
                    wavelets = self.get_wavelets(arid, logarithmic, log_after=log_after)
                    if any(wavelet is None for wavelet in wavelets):
                        continue
                    ds = f.create_dataset("/station/{}/{}".format(station, arid), data=wavelets)
                    ds.attrs["phase"] = phase
                    ds.attrs["source"] = source


                    counter += 1


if __name__ == "__main__":
    FEATURES_TINY = "data/phase/ml_features_tiny.csv"
    FEATURES = "data/phase/ml_features.csv"
    WAVEFORMS_TINY = "data/phase/ml_waveforms_tiny.csv"
    WAVEFORMS = "data/phase/ml_waveforms.csv"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-w", "--wavelets_filename", default=None,
                        help="set the path to the training dataset")
    parser.add_argument('--logarithmic', dest='logarithmic', action='store_true')
    parser.add_argument('--no-logarithmic', dest='logarithmic', action='store_false')
    parser.set_defaults(logarithmic=True)
    parser.add_argument("--log_after", dest='log_after', action='store_true',
                        help="apply log before or after wavelet function")
    parser.add_argument("--log_before", dest='log_after', action='store_false',
                        help="apply log before or after wavelet function")
    parser.set_defaults(log_after=True)

    args = parser.parse_args()

    if args.wavelets_filename is None:
        if args.logarithmic:
            wavelets_filename = "data/phase/wavelets_log.hdf5"
        else:
            wavelets_filename = "data/phase/wavelets.hdf5"
    else:
        wavelets_filename = args.wavelets_filename
    print(wavelets_filename)
    pw = PhaseWaveform(filename_features=FEATURES, filename_waveforms=WAVEFORMS_TINY)
    pw.save_wavelets(wavelets_filename, args.logarithmic, args.log_after)