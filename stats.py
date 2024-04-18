import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import csv

limit = 400

def modelLoadShedding(xs, rateFn):
    ys = []
    for x in xs:
        rate = rateFn(x)
        ys.append(rate if rate < limit else limit)
    return ys

def modelTrafficQueue(xs, rateFn):
    queue = 0
    ys = []
    qs = []
    for x in xs: # As-is, 15 sec steps
        rate = rateFn(x)
        if (rate < limit): # limit 500 -> 2 per step
            if (queue <= 0): # under the limit and no queue, add sessions
                ys.append(rate) 
            else: # if sessions queued, take from the queue until total rate is at limit
                if (rate > 0):
                    deficit = limit - rate
                    ys.append(rate + deficit)
                    queue -= (deficit / 3600) * 15.84
                else: # If rate is negative, take full limit from queue
                    ys.append(rate + limit)
                    queue -= (limit / 3600) * 15.84
        else:
            excess = rate - limit
            ys.append(limit)
            queue += (excess / 3600) * 15.84
        qs.append(queue)
    return (ys, qs)

def integrateLimitedSessionRate(xs, intFn, yInit):
    ys = [yInit]
    for i in range(0, xs.size):
        if (i + 1 < xs.size):
            total = yInit + intFn(xs[0], xs[i])
            ys.append(total if total >= 0 else 0)
    return ys

# Read CSV file
x = []
y = []
with open("data/sessions.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        x.append(row[0] / 3) # Raw data comes in 20 min segments from Datadog
        y.append(row[1])

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), tight_layout=True, sharex='all', sharey='row')

# Plot raw data
xs = np.linspace(0, 22, 5000)
spl = UnivariateSpline(x, y, s=500000)
total = math.trunc(spl.integral(0, 22))

axs[0, 0].set_title('No limiting')
axs[0, 0].set_ylabel('Sessions')
axs[2, 0].set_xlabel('Hours')
axs[0, 0].plot(xs, spl(xs), color='purple')
axs[0, 0].scatter(x, y, color='lightgrey')
axs[0, 0].text(6, 500, '~%s total sessions'%(f'{total:,}'), fontsize = 8)

# Find and plot rate (derivative)
rate = spl.derivative()

axs[1, 0].set_ylabel('Session rate (per h)')
axs[1, 0].axhline(y=0, color='whitesmoke', linestyle='-')
axs[1, 0].axhline(y=limit, color='tomato', linestyle='-')
axs[1, 0].plot(xs, rate(xs), color='green')

# Model load shedding
ys = modelLoadShedding(xs, rate)
# Integrate to model total sessions
spl2 = UnivariateSpline(xs, ys, s=500000)
i = spl2.integral
yss = integrateLimitedSessionRate(xs, i, spl(xs)[0])
spl2i = UnivariateSpline(xs, yss, s=500000)
total = math.trunc(spl2i.integral(0, 22))

axs[0, 1].set_title('Load shedding (model)')
axs[2, 1].set_xlabel('Hours')
axs[1, 1].axhline(y=0, color='whitesmoke', linestyle='-')
axs[1, 1].axhline(y=limit, color='tomato', linestyle='-')
axs[1, 1].plot(xs, ys, color='green')
axs[0, 1].plot(xs, yss, color='orange')
axs[0, 1].text(6, 500, '~%s total sessions'%(f'{total:,}'), fontsize = 8)

# Model traffic queue
(yqs, qs) = modelTrafficQueue(xs, rate)
# Integrate to model total sessions
spl3 = UnivariateSpline(xs, yqs, s=50000000)
i2 = spl3.integral
yqss = integrateLimitedSessionRate(xs, i2, spl(xs)[0])
spl3i = UnivariateSpline(xs, yqss, s=50000000)
total = math.trunc(spl3i.integral(0, 22))
spl3q = UnivariateSpline(xs, qs, s=50000000)
totalQueue = math.trunc(spl3q.integral(0, 22))

axs[0, 2].set_title('Queue (model)')
axs[2, 2].set_xlabel('Hours')
axs[1, 2].axhline(y=0, color='whitesmoke', linestyle='-')
axs[1, 2].axhline(y=limit, color='tomato', linestyle='-')
axs[1, 2].plot(xs, yqs, color='green')
axs[0, 2].plot(xs, spl(xs), color='purple', linestyle='--')
axs[0, 2].plot(xs, yss, color='orange', linestyle='--')
axs[0, 2].plot(xs, yqss)
axs[0, 2].legend(['No limiting', 'Load shedding', 'Queue'])
axs[0, 2].text(6, 500, '~%s total sessions'%(f'{total:,}'), fontsize = 8)

axs[2, 0].set_ylabel('Queue size')
axs[2, 2].plot(xs, qs, color='black')
axs[2, 2].text(12, 20, '~%s total queued'%(f'{totalQueue:,}'), fontsize = 8)

plt.show()

