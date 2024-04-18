import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

import csv

limit = 400


def modelLoadShedding(xs: np.ndarray, rateFn: UnivariateSpline) -> list:
    """
    Given a series of rates over time, applies a maximum rate by discarding rates above the limit
    """
    ys = []
    x: float
    for x in xs:
        rate: float = rateFn(x)  # type: ignore
        ys.append(rate if rate < limit else limit)
    return ys


def modelTrafficQueue(xs: np.ndarray, rateFn: UnivariateSpline) -> tuple:
    """
    Given a series of rates over time, applies a maximum rate by queueing rates above the limit
    """
    queue = 0
    ys = []
    qs = []
    for x in xs:  # As-is, 15 sec steps
        rate: float = rateFn(x)  # type: ignore
        if rate < limit:  # limit 500 -> 2 per step
            if queue <= 0:  # under the limit and no queue, add sessions
                ys.append(rate)
            else:  # if sessions queued, take from the queue until total rate is at limit
                if rate > 0:
                    deficit = limit - rate
                    ys.append(rate + deficit)
                    queue -= (deficit / 3600) * 15.84
                else:  # If rate is negative, take full limit from queue
                    ys.append(rate + limit)
                    queue -= (limit / 3600) * 15.84
        else:
            excess = rate - limit
            ys.append(limit)
            queue += (excess / 3600) * 15.84
        qs.append(queue)
    return (ys, qs)


def integrateSessionRate(xs, intFn, yInit):
    """
    Integrate the session rate to plot the number of sessions over time
    """
    ys = [yInit]
    for i in range(0, xs.size):
        if i + 1 < xs.size:
            total = yInit + intFn(xs[0], xs[i])
            ys.append(total if total >= 0 else 0)
    return ys


# Read CSV file
x = []
y = []
with open("data/sessions.csv") as csvfile:
    reader = csv.reader(
        csvfile, quoting=csv.QUOTE_NONNUMERIC
    )  # change contents to floats
    for row in reader:  # each row is a list
        x.append(float(row[0]) / 3)  # Raw data comes in 20 min segments from Datadog
        y.append(row[1])

fig, axs = plt.subplots(
    nrows=3, ncols=3, figsize=(8, 8), tight_layout=True, sharex="all", sharey="row"
)

# Plot raw data and spline fit
xs = np.linspace(0, 22, 5000)
spl = UnivariateSpline(x, y, s=500000)
total = math.trunc(spl.integral(0, 22))  # type: ignore

axs[0, 0].set_title("No limiting")
axs[0, 0].set_ylabel("Sessions")
axs[2, 0].set_xlabel("Hours")
axs[0, 0].plot(xs, spl(xs), color="purple")  # spline fit
axs[0, 0].scatter(x, y, color="lightgrey")  # raw data
axs[0, 0].text(6, 500, "~%s total sessions" % (f"{total:,}"), fontsize=8)

# Find and plot the rate (derivative)
rate = spl.derivative()

axs[1, 0].set_ylabel("Session rate (per h)")
axs[1, 0].axhline(y=0, color="whitesmoke", linestyle="-")  # zero axis
axs[1, 0].axhline(y=limit, color="tomato", linestyle="-")  # rate limit
axs[1, 0].plot(xs, rate(xs), color="green")  # rate of sessions

# Model load shedding
ys = modelLoadShedding(xs, rate)
# Integrate to model total sessions
spl2 = UnivariateSpline(xs, ys, s=500000)
i = spl2.integral
yss = integrateSessionRate(xs, i, spl(xs)[0])
spl2i = UnivariateSpline(xs, yss, s=500000)
total = math.trunc(spl2i.integral(0, 22))  # type: ignore

axs[0, 1].set_title("Load shedding (model)")
axs[2, 1].set_xlabel("Hours")
axs[1, 1].axhline(y=0, color="whitesmoke", linestyle="-")  # zero axis
axs[1, 1].axhline(y=limit, color="tomato", linestyle="-")  # rate limit
axs[1, 1].plot(xs, ys, color="green")  # rate of sessions
axs[0, 1].plot(
    xs, yss, color="orange"
)  # integral of rate (sessions over time), load shedding
axs[0, 1].text(6, 500, "~%s total sessions" % (f"{total:,}"), fontsize=8)

# Model traffic queue
(yqs, qs) = modelTrafficQueue(xs, rate)
# Integrate to model total sessions
spl3 = UnivariateSpline(xs, yqs, s=50000000)
i2 = spl3.integral
yqss = integrateSessionRate(xs, i2, spl(xs)[0])
spl3i = UnivariateSpline(xs, yqss, s=50000000)
total = math.trunc(spl3i.integral(0, 22))  # type: ignore
spl3q = UnivariateSpline(xs, qs, s=50000000)
totalQueue = math.trunc(spl3q.integral(0, 22))  # type: ignore

axs[0, 2].set_title("Queue (model)")
axs[2, 2].set_xlabel("Hours")
axs[1, 2].axhline(y=0, color="whitesmoke", linestyle="-")  # zero axis
axs[1, 2].axhline(y=limit, color="tomato", linestyle="-")  # rate limit
axs[1, 2].plot(xs, yqs, color="green")  # rate of sessions
axs[0, 2].plot(
    xs, spl(xs), color="purple", linestyle="--"
)  # original spline fit, dashed
axs[0, 2].plot(xs, yss, color="orange", linestyle="--")  # load shedding, dashed
axs[0, 2].plot(xs, yqss)  # integral of rate (sessions over time), queued
axs[0, 2].legend(["No limiting", "Load shedding", "Queue"])
axs[0, 2].text(6, 500, "~%s total sessions" % (f"{total:,}"), fontsize=8)

axs[2, 0].set_ylabel("Queue size")
axs[2, 2].plot(xs, qs, color="black")  # size of queue over time
axs[2, 2].text(12, 20, "~%s total queued" % (f"{totalQueue:,}"), fontsize=8)

plt.show()
