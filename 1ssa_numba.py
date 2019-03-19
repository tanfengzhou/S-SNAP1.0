import pickle
import numpy as np
import math
from numba import jit

sr=500      # Hz
win=1       # seconds
step=0.5    # seconds
nsta = 69   # number of stations

with open('ptraveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)

ptraveltimes = comein[0]

with open('straveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)

straveltimes = comein[0]

st=[]
n=0
for i in st:
    with open('./H3/data/trace' + str(n) + '.p', 'rb') as f:
        comein=pickle.load(f)
    st.append(comein[0])
    n=n+1


##############################################################

# vn and hn are normalized seismograms

vn=[]
for i in range(0,69):
    vv=[]
    for j in range(0,60):
        data = st[j*69*3+i].data
        data = data-np.mean(data)
        data = np.abs(data)
        if math.isnan(np.median(data)) == True or np.median(data) == 0:
            data = np.ones(len(data))
        else:
            data = data/np.median(data)
        data = data ** (1/3)
        vv.append(data)

    vn.append(np.concatenate(vv))

hn = []
for i,ii in zip(range(69, 69*2),range(69*2, 69*3)):
    vv=[]
    for j in range(0, 60):
        data1 = st[j * 69 * 3 + i].data
        data1 = data1 - np.mean(data1)
        data2 = st[j * 69 * 3 + ii].data
        data2 = data2 - np.mean(data2)
        data = (data1**2 + data2**2) ** 0.5
        if math.isnan(np.median(data)) == True or np.median(data) == 0:
            data = np.ones(len(data))
        else:
            data = data/np.median(data)
        data = data ** (1/3)
        vv.append(data)

    hn.append(np.concatenate(vv))


@jit(nopython=True)
def scan(seismograms, timelags, saferest, sr, win, step):
    # saferest=max(timelags)
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    for index, i in enumerate(t):
        flag = timelags + i
        stack = np.zeros(round(win * sr))
        n = -1
        for j in range(seismograms.shape[0]):
            n = n + 1
            stack = stack + seismograms[j][int(round((flag[n] - win / 4) * sr)):int(round((flag[n] - win / 4) * sr + win * sr))]
        br[index] = (stack ** 2).sum().item()

    return br


@jit(nopython=True)
def scanmax(seismograms, timelags, saferest, sr, win, step):
    # saferest=max(timelags)
    t = np.arange(win / 2, len(seismograms[0]) / sr - saferest - win, step)
    br = np.zeros(t.shape)
    for index, i in enumerate(t):
        flag = timelags + i
        stack = 0
        n = -1
        for j in range(seismograms.shape[0]):
            n = n + 1
            stack = stack + np.max(seismograms[j][int(round((flag[n] - win / 4) * sr)):int(round((flag[n] - win / 4) * sr + win * sr))])
        br[index] = stack

    return br


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def kur_scan(timelags, vn, sr, win, step):
    saferest = np.max(timelags)
    brshape = scanmax(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    for i in range(timelags.shape[0]):
        br[i] = scanmax(seismograms=vn, timelags=timelags[i], saferest=saferest, sr=sr, win=win, step=step)
        if i % 100 == 0:
            print(i)
    return br


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def med_scan(timelags, vn, sr, win, step):
    saferest = np.max(timelags)
    brshape = scan(seismograms=vn, timelags=timelags[0], saferest=saferest, sr=sr, win=win, step=step).shape
    br = np.zeros((timelags.shape[0], brshape[0]))
    for i in range(timelags.shape[0]):
        br[i] = scan(seismograms=vn, timelags=timelags[i], saferest=saferest, sr=sr, win=win, step=step)
        if i % 100 == 0:
            print(i)
    return br

brp = med_scan(np.array(ptraveltimes), np.array(vn), sr, win, step)
brp_med=((brp/(win*sr))**0.5/nsta) ** 3

brs = med_scan(np.array(straveltimes), np.array(hn), sr, win, step)
brs_med=((brs/(win*sr))**0.5/nsta) ** 3

pickle.dump([brs_med],open('./H3/ssabrmap_S.p','wb'))
pickle.dump([brp_med],open('./H3/ssabrmap_P.p','wb'))










