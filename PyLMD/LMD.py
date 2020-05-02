import numpy as np
from scipy.signal import argrelextrema


class LMD():
    """
    .. _LMD:

    **Local Mean Decomposition**

    Method of decomposing signal into Product Functions (PFs)
    based on algorithm presented in Jonathan S. Smith.[1]

    Modified from this post (this package is vectorized LMD algorithm using numpy):
        声振论坛 陈波 [其他]LMD算法Python程序[2]

    Parameters
    ----------
    INCLUDE_ENDPOINTS : bool, (default: True)
        Whether to treat the endpoint of the signal as a pseudo-extreme point
    max_smooth_iteration : int, (default: 12)
        Maximum number of iterations of moving average algorithm.
    max_envelope_iteration : int, (default: 200)
        Maximum number of iterations when separating local envelope signals.
    envelope_epsilon : float, (default: 0.01)
        Terminate processing when obtaining pure FM signal.
    convergence_epsilon : float, (default: 0.01)
        Terminate processing when modulation signal converges.
    max_num_pf : int, (default: 8)
        The maximum number of PFs generated.

    Return
    ----------
    PFs: numpy array
        The decompose functions arrange is arranged from high frequency to low frequency.
    residue: numpy array
        residual component


    References
    ----------

    [1] Jonathan S. Smith. The local mean decomposition and its application to EEG
    perception data. Journal of the Royal Society Interface, 2005, 2(5):443-454
    [2] LMD算法Python程序 http://forum.vibunion.com/thread-138071-1-1.html

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 100, 101)
    >>> y = 2 / 3 * np.sin(x * 30) + 2 / 3 * np.sin(x * 17.5) + 4 / 5 * np.cos(x * 2)
    >>> lmd = LMD()
    >>> PFs, resdue = lmd.lmd(S)
    >>> PFs.shape
    (6, 101)
    """

    def __init__(self, include_endpoints=True, max_smooth_iteration=12, max_envelope_iteration=200, envelope_epsilon=0.01, convergence_epsilon=0.01, max_num_pf=8):

        self.INCLUDE_ENDPOINTS = include_endpoints
        self.MAX_SMOOTH_ITERATION = max_smooth_iteration
        self.MAX_ENVELOPE_ITERATION = max_envelope_iteration
        self.ENVELOPE_EPSILON = envelope_epsilon
        self.CONVERGENCE_EPSILON = convergence_epsilon
        self.MAX_NUM_PF = max_num_pf

    def is_monotonous(self, signal):
        """Determine whether the signal is a (non-strict) monotone sequence"""

        def is_monotonous_increase(signal):
            y0 = signal[0]
            for y1 in signal:
                if y1 < y0:
                    return False
                y0 = y1
            return True

        def is_monotonous_decrease(signal):
            y0 = signal[0]
            for y1 in signal:
                if y1 > y0:
                    return False
                y0 = y1
            return True

        if len(signal) <= 0:
            return True
        else:
            return is_monotonous_increase(signal) \
                or is_monotonous_decrease(signal)

    def find_extrema(self, signal):
        """Find all local extreme points of the signal"""

        n = len(signal)

        extrema = np.append(argrelextrema(signal, np.greater)[0], argrelextrema(signal, np.less)[0])
        extrema.sort()
        if self.INCLUDE_ENDPOINTS:
            if extrema[0] != 0:
                extrema = np.insert(extrema, 0, 0)
            if extrema[-1] != n-1:
                extrema = np.append(extrema, n-1)

        return extrema

    def moving_average_smooth(self, signal, window):
        n = len(signal)

        # at least one nearby sample is needed for average
        if window < 3:
            window = 3

        # adjust length of sliding window to an odd number for symmetry
        if (window % 2) == 0:
            window += 1

        half = window // 2

        weight = np.array(list(range(1, half+2)) + list(range(half, 0, -1)))
        assert (len(weight) == window)

        def is_smooth(signal):
            for x in range(1, n):
                if signal[x] == signal[x-1]:
                    return False
            return True

        smoothed = signal
        for _ in range(self.MAX_SMOOTH_ITERATION):
            head = list()
            tail = list()
            w_num = half
            for i in range(half):
                head.append(np.array([smoothed[j] for j in range(i - (half - w_num), i + half + 1)]))
                tail.append(np.flip([smoothed[-(j + 1)] for j in range(i - (half - w_num), i + half + 1)]))
                w_num -= 1

            smoothed = np.convolve(smoothed, weight, mode='same')
            smoothed[half: - half] = smoothed[half: - half] / sum(weight)

            w_num = half
            for i in range(half):
                smoothed[i] = sum(head[i] * weight[w_num:]) / sum(weight[w_num:])
                smoothed[-(i + 1)] = sum(tail[i] * weight[: - w_num]) / sum(weight[: - w_num])
                w_num -= 1
            if is_smooth(smoothed):
                break
        return smoothed

    def local_mean_and_envelope(self, signal, extrema):
        """Calculate the local mean function and local envelope function according to the location of the extreme points."""
        n = len(signal)
        k = len(extrema)
        assert(1 < k <= n)
        # construct square signal
        mean = []
        enve = []
        prev_mean = (signal[extrema[0]] + signal[extrema[1]]) / 2
        prev_enve = abs(signal[extrema[0]] - signal[extrema[1]]) / 2
        e = 1
        for x in range(n):
            if (x == extrema[e]) and (e + 1 < k):
                next_mean = (signal[extrema[e]] + signal[extrema[e+1]]) / 2
                mean.append((prev_mean + next_mean) / 2)
                prev_mean = next_mean
                next_enve = abs(signal[extrema[e]] - signal[extrema[e+1]]) / 2
                enve.append((prev_enve + next_enve) / 2)
                prev_enve = next_enve
                e += 1
            else:
                mean.append(prev_mean)
                enve.append(prev_enve)
        # smooth square signal
        window = max(np.diff(extrema)) // 3
        return np.array(mean), self.moving_average_smooth(mean, window), \
            np.array(enve), self.moving_average_smooth(enve, window)

    def extract_product_function(self, signal):
        s = signal
        n = len(signal)
        envelopes = []

        def component():
            c = s
            for e in envelopes:  # Caculate PF，using PF_i(t) = a_i(t)* s_in()，其中a_i = a_i0 * a_i1 * ... * a_in
                c = c * e
            return c

        for _ in range(self.MAX_ENVELOPE_ITERATION):
            extrema = self.find_extrema(s)
            if len(extrema) <= 3:
                break
            _m0, m, _a0, a = self.local_mean_and_envelope(s, extrema)
            for i in range(len(a)):
                if a[i] <= 0:
                    a[i] = 1 - 1e-4

            #　subtracted from the original data.
            h = s - m
            #  amplitude demodulated by dividing a.
            t = h / a

            # Terminate processing when obtaining pure FM signal.
            err = sum(abs(1-a)) / n
            if err <= self.ENVELOPE_EPSILON:
                break
            # Terminate processing when modulation signal converges.
            err = sum(abs(s-t)) / n
            if err <= self.CONVERGENCE_EPSILON:
                break
            envelopes.append(a)
            s = t

        return component()

    def lmd(self, signal):
        pf = []
        # until the residual function is close to a monotone function
        residue = signal[:]
        while (len(pf) < self.MAX_NUM_PF) and \
            (not self.is_monotonous(residue)) and \
                (len(self.find_extrema(residue)) >= 5):
            component = self.extract_product_function(residue)
            residue = residue - component
            pf.append(component)
        return np.array(pf), residue


if __name__ == "__main__":
    lmd = LMD()
    x = np.linspace(0, 100, 101)
    y = 2 / 3 * np.sin(x * 30) + 2 / 3 * np.sin(x * 17.5) + 4 / 5 * np.cos(x * 2)
    PFs, res = lmd.lmd(y)
    print(PFs.shape)

    '''
    import matplotlib.pyplot as plt
    plt.subplot(8, 1, 1)
    plt.title('original signal')
    plt.plot(x, y)
    for i in range(1, 7):
        plt.subplot(8, 1, i+1)
        plt.title('PF%d' % i)
        plt.plot(x, PFs[i - 1])
    plt.subplot(8, 1, 8)
    plt.title('residue')
    plt.plot(x, res)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    '''
