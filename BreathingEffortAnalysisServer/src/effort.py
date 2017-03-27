import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.signal


def interpolate(x: np.ndarray, y: np.ndarray, new_x: np.ndarray):
    return sp.interpolate.interp1d(x, y, bounds_error=False, fill_value='extrapolate')(new_x)


def envelope(signal: np.ndarray, x: np.ndarray = None, top=True, iterations=1):
    if x is None:
        x = np.arange(len(signal))
    comparator = np.greater if top else np.less

    for i in range(iterations):
        # noinspection PyUnresolvedReferences
        local_extrema = sp.signal.argrelextrema(signal, comparator)[0]
        signal = interpolate(local_extrema, signal[local_extrema], x)

    return signal


def rms(signal: np.ndarray):
    return np.sqrt(np.mean(signal ** 2))


def mad(data: np.ndarray):
    data_median = np.median(data)
    return np.median(np.abs(data - data_median))


class EffortCalculator:
    def effort(self, data: np.ndarray):
        raise NotImplementedError


class EffortPSD(EffortCalculator):
    def effort(self, data: np.ndarray):
        psd_x, psd_x = sp.signal.welch(data)
        return np.trapz(psd_x, psd_x)


class EffortBaselineModulation(EffortCalculator):
    def effort(self, data: np.ndarray):
        baseline = envelope(data, iterations=1, top=False)
        return mad(baseline)


class EffortAmplitudeModulation(EffortCalculator):
    def effort(self, data: np.ndarray):
        top = envelope(data, iterations=1, top=True)
        bottom = envelope(data, iterations=1, top=False)
        return mad(top) - mad(bottom)


class EffortDerivative(EffortCalculator):
    def effort(self, data: np.ndarray):
        top = envelope(np.diff(data), iterations=2, top=True)
        return mad(top)


class EffortHeartRate(EffortCalculator):
    # noinspection PyUnresolvedReferences
    def effort(self, data: np.ndarray):
        mins = sp.signal.argrelmin(data)[0]
        minsMins = sp.signal.argrelmin(data[mins])[0]
        ibi = np.diff(mins[minsMins])  # samples per beat
        hr = 1 / ibi  # beats per sample
        return mad(hr)


class EffortFused(EffortCalculator):
    def __init__(self):
        self.otherCalculators = [
            EffortPSD(),
            EffortBaselineModulation(),
            EffortAmplitudeModulation(),
            EffortDerivative()
        ]

    def effort(self, data: np.ndarray):
        efforts = [calc.effort(data) for calc in self.otherCalculators]
        return np.prod(efforts)


class EffortQueue:
    def __init__(self, win_size: int, calculator: EffortCalculator):
        self.win_size = win_size
        self.__samples = np.zeros(win_size)
        self.__effort = np.zeros(win_size)
        self.total_samples = 0
        self.calculator = calculator

    def feed(self, samples: np.ndarray):
        num_samples = len(samples)
        if self.total_samples < self.win_size:
            self.total_samples += num_samples // 2

        self.__samples = np.roll(self.__samples, -num_samples)
        self.__samples[-num_samples:] = samples

        latest_effort = self.calculator.effort(self.samples())
        effort_interp = np.linspace(self.__effort[-1], latest_effort, num_samples)
        self.__effort = np.roll(self.__effort, -num_samples)
        self.__effort[-num_samples:] = effort_interp

        return latest_effort

    def samples(self):
        return self.__samples[-self.total_samples:]

    def effort(self):
        return self.__effort[-self.total_samples:]

    def clear(self):
        self.__samples = np.zeros(self.win_size)
        self.__effort = np.zeros(self.win_size)
        self.total_samples = 0
