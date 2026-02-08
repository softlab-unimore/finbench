from collections import deque

import numpy as np


class Rolling:
    """ 1-D array rolling """

    def __init__(self, window):
        self.window = window
        self.na_count = window
        self.barv = deque([float('nan')] * window)

    def update(self, val):
        pass


class Slope(Rolling):
    """ 1-D array rolling slope """

    def __init__(self, window):
        super(Slope, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.xy_sum = 0

    def update(self, val):
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        _val = self.barv[0]
        if not np.isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1

        self.barv.popleft()

        if np.isnan(val):
            self.na_count += 1
            # return float('nan')
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.xy_sum += self.window * val

        N = self.window - self.na_count

        # Avoid division by zero
        denominator = (N * self.x2_sum - self.x_sum * self.x_sum)
        if denominator == 0:
            return float('nan')

        return (N * self.xy_sum - self.x_sum * self.y_sum) / denominator


class Resi(Rolling):
    """1-D array rolling residuals"""

    def __init__(self, window):
        super(Resi, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.xy_sum = 0

    def update(self, val):
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        _val = self.barv[0]
        if not np.isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1

        self.barv.popleft()

        if np.isnan(val):
            self.na_count += 1
            return float('nan')
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.xy_sum += self.window * val

        N = self.window - self.na_count

        if N * self.x2_sum - self.x_sum * self.x_sum == 0:
            return float('nan')

        slope = (N * self.xy_sum - self.x_sum * self.y_sum) / (N * self.x2_sum - self.x_sum * self.x_sum)
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope * x_mean

        return val - (slope * self.window + interp)


class Rsquare(Rolling):
    """ 1-D array rolling rsquare """

    def __init__(self, window):
        super(Rsquare, self).__init__(window)
        self.i_sum = 0
        self.x_sum = 0
        self.x2_sum = 0
        self.y_sum = 0
        self.y2_sum = 0
        self.xy_sum = 0

    def update(self, val):
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum

        _val = self.barv[0]
        if not np.isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
            self.y2_sum -= _val * _val
        else:
            self.na_count -= 1

        self.barv.popleft()

        if np.isnan(val):
            self.na_count += 1
            # return float('nan')
        else:
            self.i_sum += 1
            self.x_sum += self.window
            self.x2_sum += self.window * self.window
            self.y_sum += val
            self.y2_sum += val * val
            self.xy_sum += self.window * val

        N = self.window - self.na_count

        if N * self.x2_sum - self.x_sum * self.x_sum <= 0 or N * self.y2_sum - self.y_sum * self.y_sum <= 0:
            return float('nan')

        rvalue = (N * self.xy_sum - self.x_sum * self.y_sum) / np.sqrt(
            (N * self.x2_sum - self.x_sum * self.x_sum) * (N * self.y2_sum - self.y_sum * self.y_sum))

        return rvalue * rvalue


def rolling(r, a):
    """ Apply rolling calculation to array a using calculator r """
    N = len(a)
    ret = np.empty(N)
    for i in range(N):
        ret[i] = r.update(a[i])
    return ret


def rolling_slope(a, window):
    """ Calculate rolling slope for array a with given window """
    r = Slope(window)
    return rolling(r, a)


def rolling_rsquare(a, window):
    """ Calculate rolling rquare for array a with given window """
    r = Rsquare(window)
    return rolling(r, a)


def rolling_resi(a, window):
    """ Calculate rolling resi for array a with given window """
    r = Resi(window)
    return rolling(r, a)
