def mag(data, delta, dis, geo_spr=-2):
    npts = len(data)
    fs = 1/delta
    import numpy as np
    time = np.arange(0, npts*delta, delta)
    Y = np.fft.fft(data * 0.001 * 100)      # mV to V, m/s to cm/s
    f = np.arange(0, fs / 2, 1 / (npts * delta))

    f0 = 10
    lamda = 1 / (2 ** 0.5)
    H=[]
    for i in range(0, len(f)):
        h = 85.8 #* (f[i] / f0) ** 2 / (-(f[i] / f0) ** 2 + 2j * lamda * f[i] / f0 + 1)
        H.append(h)
        Y[i] = Y[i]/h

    H = np.array(H)
    Habs = abs(H)
    #plt.loglog(f, Habs)
    #plt.show()

    P2 = abs(Y/npts)      # m/s to cm/s

    P1 = P2[0:len(f)]
    P1[1:len(P1)] = P1[1:len(P1)] * 2

    hypo_dist = dis
    log_src_data = np.log10(P1) - (geo_spr * np.log10(hypo_dist))

    source_DISP = 10 ** log_src_data /((2*np.pi*f)**1)

    source_ACCN = source_DISP*((2*np.pi*f)**2)

    '''
    import matplotlib.pyplot as plt
    plt.loglog(f, source_DISP)
    plt.show()

    plt.clf()
    plt.loglog(f, source_ACCN)
    plt.show()
    '''

    return ([source_DISP, source_ACCN])


#########################################################################################################

import pickle
import numpy as np
#import scipy.signal as ss
#import matplotlib.pyplot as plt
import getbeta

sr=500
win=1
step=0.5
nsta=69

rho=2.7

r0=1
rp=0.55
v=0.71
F=2
cf=[0.5,20]

readdir = './H3/'
savedir = './H3/'

npts = win*sr
fs = sr
delta = 1/fs
fff = np.arange(0, fs/2, 1/(npts * delta))


with open(readdir + 'cataloghigh.p' , 'rb') as f:
    comein=pickle.load(f)

catalog=comein[0]

with open(readdir + 'MAXI_PRED.p' , 'rb') as f:
    comein=pickle.load(f)

events=comein[0]

with open(readdir + 'ponsets.p' , 'rb') as f:
    comein=pickle.load(f)

ponsets=comein[0]

with open(readdir + 's1onsets.p' , 'rb') as f:
    comein=pickle.load(f)

s1onsets=comein[0]

with open(readdir + 's2onsets.p' , 'rb') as f:
    comein=pickle.load(f)

s2onsets=comein[0]


stz=[]
sty=[]
stx=[]
readin = open('stz','r')
lines = readin.readlines()
for i in lines:
    stz.append(float(i))

readin = open('sty','r')
lines = readin.readlines()
for i in lines:
    sty.append(float(i))

readin = open('stx','r')
lines = readin.readlines()
for i in lines:
    stx.append(float(i))

st=[]
n=0
for i in range(0,12420):
    with open('./H3/data/trace' + str(n) + '.p', 'rb') as f:
        comein=pickle.load(f)
    st.append(comein[0])
    n=n+1

vertical=[]
for i in range(0,69):
    vv=[]
    for j in range(0,60):
        vv.append(st[j*69*3+i].data)

    vertical.append(np.concatenate(vv))

horizon1 = []
for i in range(69, 69*2):
    vv=[]
    for j in range(0, 60):
        vv.append(st[j * 69 * 3 + i].data)

    horizon1.append(np.concatenate(vv))

horizon2 = []
for i in range(69*2, 69*3):
    vv=[]
    for j in range(0, 60):
        vv.append(st[j * 69 * 3 + i].data)

    horizon2.append(np.concatenate(vv))

eventnum=-1
for i in range(0, len(catalog)):
    eventnum = eventnum+1
    eventtime=catalog[i][0]
    evla=catalog[i][1]
    evlo=catalog[i][2]
    evdep=catalog[i][3]
    while events[eventnum][0] < 0:
        eventnum = eventnum+1

    p = ponsets[eventnum]
    s1 = s1onsets[eventnum]
    s2 = s2onsets[eventnum]

    fdis = []
    facc = []
    stnum=-1
    for j in p:
        stnum=stnum+1
        if j<0:
            continue
        else:
            data = vertical[stnum][int((j-win/4)*sr):int((j+3*win/4)*sr)]
            hypo_dis = ((evla - sty[stnum])**2 + (evlo-stx[stnum]) **2 + ((evdep+stz[stnum])/1000)**2) ** 0.5
            [a, b] = mag(data=data, delta=1/sr, dis=hypo_dis)

        fdis.append(a)
        facc.append(b)

    stnum = -1
    for j in s1:
        stnum = stnum + 1
        if j < 0:
            continue
        else:
            data = horizon1[stnum][int((j - win / 4) * sr):int((j + 3 * win / 4) * sr)]
            hypo_dis = ((evla - sty[stnum])**2 + (evlo-stx[stnum]) **2 + ((evdep+stz[stnum])/1000)**2) ** 0.5
            [a, b] = mag(data=data, delta=1 / sr, dis=hypo_dis)

        fdis.append(a)
        facc.append(b)

    stnum = -1
    for j in s2:
        stnum = stnum + 1
        if j < 0:
            continue
        else:
            data = horizon2[stnum][int((j - win / 4) * sr):int((j + 3 * win / 4) * sr)]
            hypo_dis = ((evla - sty[stnum])**2 + (evlo-stx[stnum]) **2 + ((evdep+stz[stnum])/1000)**2) ** 0.5
            [a, b] = mag(data=data, delta=1 / sr, dis=hypo_dis)

        fdis.append(a)
        facc.append(b)

    fdis = np.array(fdis)
    facc = np.array(facc)

    '''
    plt.clf()
    for pp in fdis:
        plt.loglog(fff, pp)
    plt.show()

    plt.clf()
    for pp in facc:
        plt.loglog(fff, pp)
    plt.show()
    '''

    FDISP = np.median(fdis, 0)
    FACCN = np.median(facc, 0)

    #FDISP = ss.savgol_filter(FDISP, 15, 2)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.loglog(fff, FDISP, picker=3)
    '''
    '''
    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        global points
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)
        plt.close()


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    pointout = np.mean(np.array(points),0)
    '''
    beta = getbeta.beta(evdep)

    FD = FDISP[1]
    #FD = pointout[1]
    #print(FD)
    C = ((rp*v*F)/(4*np.pi*rho*beta**3*r0))*10**(-20)
    M0 = FD/C
    Mw = 2/3*np.log10(M0)-10.71
    print('Mw = '+str(Mw))
    catalog[i].append(Mw)
#    plt.savefig(savedir + 'event'+str(i)+'.pdf')
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.loglog(fff, FACCN, picker=3)
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    pointout = np.mean(np.array(points), 0)

    fc = pointout[0]
    stressdrop = M0 * (fc/(4.9*10**6*beta))**3
    print("stress drop = "+str(stressdrop))
    catalog[i].append(stressdrop)
    '''

pickle.dump([catalog],open(savedir + 'magnitudeauto.p','wb'))







