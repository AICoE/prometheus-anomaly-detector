import numpy as np
from scipy.stats import norm

class Accumulator:
    def __init__(self,thresh):
        self._counter = 0
        self.thresh = thresh
    def inc(self, val):
        self._counter += val
    def count(self):
        return self._counter

def detect_anomalies(predictions, data):
    if len(predictions) != len(data) :
        raise IndexError

    # parameters
    lower_bound_thresh = predictions["yhat_lower"].min()
    upper_bound_thresh = predictions["yhat_upper"].max()
    diff_thresh = 3*data["y"].std()
    acc_thresh = int(0.1*np.shape(predictions)[0])
    epsilon = .01

    diffs = []
    acc = Accumulator(acc_thresh)
    preds = np.array(predictions["yhat"])
    dat = np.array(data["y"])
    for i in range(0, np.shape(predictions)[0]):
        diff = preds[i] - dat[i]
        if abs(diff) > diff_thresh:
            # upper bound anomaly, increment counter
            acc.inc(1)
        elif dat[i] < lower_bound_thresh:
            # found trough, decrement so that acc will decay to 0
            acc.inc(-3)
        elif dat[i] > upper_bound_thresh:
            # found peak, decrement so that acc will decay to 0
            acc.inc(-3)
        else:
            # no anomaly, decrement by 2
            acc.inc(-2)

        diffs.append(max(diff, 0))

    if acc.count() > acc.thresh:
        acc_anomaly = True
    else:
        acc_anomaly = False
    w_size = int(0.8*len(data))
    w_prime_size = len(data) - w_size

    w = diffs[0:w_size]
    w_prime = diffs[w_size:]

    w_mu = np.mean(w)
    w_std = np.std(w)
    w_prime_mu = np.mean(w_prime)

    if w_std == 0:
        L_t = 0
    else:
        L_t = 1 - norm.sf((w_prime_mu - w_mu)/w_std)

    print(L_t)
    if L_t >= 1 - epsilon:
        tail_prob_anomaly = True
    else:
        tail_prob_anomaly = False

    return acc_anomaly and tail_prob_anomaly
