import pickle
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import stations_plot
from numba import jit

@jit(nopython=True)
def mykurtosis(X):
    if np.var(X) == 0 or (np.var(X)**2) == 0:
        return -3.0

    m4 = 0
    mu = np.mean(X)
    for i in range(len(X)):
        m4 += (X[i] - mu)**4

    m4 = m4/len(X)
    return m4/(np.var(X)**2) - 3

dirread='./H3/'
dirsave='./H3/'

sr=500
win=1
step=0.5
nsta=69

with open(dirread + 'ssabrmap_S.p' , 'rb') as f:
    comein=pickle.load(f)
brmaps=np.array(comein[0])

shape_s=np.shape(brmaps)

with open(dirread + 'ssabrmap_P.p' , 'rb') as f:
    comein=pickle.load(f)
brmapp=np.array(comein[0])

brmapp = brmapp[:,0:shape_s[1]]

brmap=np.multiply(brmaps,brmapp)
brmax=[]
for i in range(0,len(brmap[0])):
    brmax.append(max(brmap[:,i]))

plt.plot(brmax)
plt.yscale('log')

plt.savefig(dirsave + 'p&s1hourlog.pdf')

#######################################################################
# to see if seismograms are aligned

with open('ptraveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)

ptraveltimes = comein[0]

with open('straveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)
straveltimes = comein[0]

from detect_peaks import detect_peaks
peaktimes = detect_peaks(x=brmax, mph=1, mpd=0)

print(peaktimes)

with open('./H3/H3data.p', 'rb' ) as f:
    comein=pickle.load(f)
    
st = comein[0]

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

horizon1=np.array(horizon1)
horizon2=np.array(horizon2)
vertical=np.array(vertical)

h1m=[]
for i in horizon1:
    a = np.mean(i)
    h1m.append(i-a)

h2m=[]
for i in horizon2:
    a=np.mean(i)
    h2m.append(i-a)

vm=[]
for i in vertical:
    a=np.mean(i)
    vm.append(i-a)

hn=[]
for i,j in zip(h1m,h2m):
    b=(i**2+j**2)**0.5
    a=max(b)
    hn.append(b/a)

vn=[]
for i in vm:
    b=np.abs(i)
    a=max(b)
    vn.append(b/a)

# put time lags here and see the seismograms.
peaks=[]
for i in peaktimes:
    peaks.append(brmap[:,i])

pp=[]
for i in peaks:
    m=max(i)
    p=[j for j,k in enumerate(i) if k==m]
    pp.append(p)

print(pp)

########################################################################
# plot ssa location results for reference

with open('studygrids.p' , 'rb') as  f:
    comein=pickle.load(f)

studygrids=comein[0]

evlas=[]
evlos=[]
evdeps=[]
for i in pp:
    evlas.append(studygrids[i[0]][0])
    evlos.append(studygrids[i[0]][1])
    evdeps.append(studygrids[i[0]][2])

print(evlas)
print(evlos)
print(evdeps)
pickle.dump([pp,evlas,evlos,evdeps],open(dirsave + 'SSAlocation.p','wb'))
###############################################################################################
# onset time picking

ponsets=[]
number=0
print("There are "+str(len(peaktimes))+"events.\n")
for i,nm in zip(peaktimes,pp):
    number=number+1
    print('event'+str(number)+'p')
    plt.clf()
    k=0
    on=[]

    #fig = plt.figure(figsize=(11, 11))
    #fff = fig.add_subplot(111)

    flag=np.array(ptraveltimes[nm[0]])+i*step+win/2    # flag is real time on the seismogram, referring the beginning of seismogram
    for j,m,n in zip(vertical,horizon1,horizon2):
        k=k+1
        a=j[int(round((flag[k-1]-9*win/4)*sr)):int(round((flag[k-1]-win/4)*sr+3*win*sr))]

        standard = len(a) / 2 - win / 4 * sr
        potential = range(int(standard - win / 2 * sr), int(standard + win / 2 * sr))

        kur=[]
        for potentialtime in potential:
            array=a[int(potentialtime-win*sr):potentialtime]
            kur.append(mykurtosis(array))

        kur=np.array(kur)

        # calculate kurtosis rate

        kurrate=[]
        maxkr=0
        for ll in range(0,len(kur)-5):
            rate=kur[ll+5]-kur[ll]
            if rate>maxkr:
                maxkr=rate
                maxlocation=ll
            kurrate.append(rate)

        kur_threshold=3
        kur_threshold2=1
        for ll in range(0,len(kurrate)):
            if kurrate[ll] > kur_threshold:
                onsetflag=ll
                break
            if ll == len(kurrate)-1:
                if maxkr>kur_threshold2:
                    onsetflag=maxlocation-10  # -10 is from experience
                else:
                    onsetflag=-1

        timedelay=(potential[onsetflag]-standard)/sr
        arrive=flag[k-1]+timedelay
        if onsetflag<0:
            arrive=-1
        on.append(arrive)

        #plt.plot(kur/max(kur)+k)

        #if onsetflag > 0:
        #    plt.plot(a/max(abs(a))+2*k, linewidth=0.7, marker='|', linestyle='-', markeredgecolor='black', ms=5, markevery=[potential[onsetflag]])
        #else:
        #    plt.plot(a/ max(abs(a)) + 2 * k, linewidth=0.7, ls='-')

    #fff.set_aspect(30)
    #plt.savefig(dirsave+str(i)+'p5.eps')

    ponsets.append(on)

ponsets=np.array(ponsets)
print(ponsets)
pickle.dump([ponsets],open(dirsave + 'ponsets.p','wb'))


s1onsets=[]
n=0
for i,nm in zip(peaktimes,pp):
    n=n+1
    print('event'+str(n)+'s1')
    plt.clf()
    k=0
    on=[]

    #fig = plt.figure(figsize=(11, 11))
    #fff = fig.add_subplot(111)
    
    flag=np.array(straveltimes[nm[0]])+i*step+win/2    # flag is real time on the seismogram, referring the beginning of seismogram
    for j in horizon1:
        k=k+1
        a=j[int(round((flag[k-1]-9*win/4)*sr)):int(round((flag[k-1]-win/4)*sr+3*win*sr))]
        standard=len(a)/2-win/4*sr
        potential=range(int(standard-win/2*sr),int(standard+win/2*sr))
        kur=[]
        for potentialtime in potential:
            array=a[potentialtime-win*sr:potentialtime]
            kur.append(mykurtosis(array))

        kur=np.array(kur)

        kurrate = []
        maxkr = 0
        for ll in range(0, len(kur) - 5):
            rate = kur[ll + 5] - kur[ll]
            if rate > maxkr:
                maxkr = rate
                maxlocation = ll
            kurrate.append(rate)

        kur_threshold = 3
        kur_threshold2 = 1
        for ll in range(0, len(kurrate)):
            if kurrate[ll] > kur_threshold:
                onsetflag = ll
                break
            if ll == len(kurrate) - 1:
                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

        timedelay = (potential[onsetflag] - standard) / sr
        arrive = flag[k - 1] + timedelay
        if onsetflag < 0:
            arrive = -1
        on.append(arrive)

        # plt.plot(kur/max(kur)+k)
        '''
        if onsetflag > 0:
            plt.plot(a / max(abs(a)) + 2 * k,linewidth=0.7, marker= '|', ls='-', markeredgecolor='black', ms=5,
                     markevery=[potential[onsetflag]])
        else:
            plt.plot(a / max(abs(a)) + 2 * k,linewidth=0.7, ls='-')
        '''
            
    #fff.set_aspect(30)
    #plt.show()
    #plt.savefig(dirsave + str(i) + 's1.eps')

    s1onsets.append(on)

s1onsets=np.array(s1onsets)
print(s1onsets)
pickle.dump([s1onsets],open(dirsave + 's1onsets.p','wb'))


s2onsets=[]
n=0
for i,nm in zip(peaktimes,pp):
    n=n+1
    print('event'+str(n)+'s2')
    plt.clf()
    k=0
    on=[]

    #fig = plt.figure(figsize=(11, 11))
    #fff = fig.add_subplot(111)

    flag=np.array(straveltimes[nm[0]])+i*step+win/2    # flag is real time on the seismogram, referring the beginning of seismogram
    for j in horizon2:
        k=k+1
        a=j[int(round((flag[k-1]-9*win/4)*sr)):int(round((flag[k-1]-win/4)*sr+3*win*sr))]
        standard=len(a)/2-win/4*sr
        potential=range(int(standard-win/2*sr),int(standard+win/2*sr))
        kur=[]
        for potentialtime in potential:
            array=a[potentialtime-win*sr:potentialtime]
            kur.append(mykurtosis(array))

        kur=np.array(kur)

        kurrate = []
        maxkr = 0
        for ll in range(0, len(kur) - 5):
            rate = kur[ll + 5] - kur[ll]
            if rate > maxkr:
                maxkr = rate
                maxlocation = ll
            kurrate.append(rate)

        kur_threshold = 3
        kur_threshold2 = 1
        for ll in range(0, len(kurrate)):
            if kurrate[ll] > kur_threshold:
                onsetflag = ll
                break
            if ll == len(kurrate) - 1:
                if maxkr > kur_threshold2:
                    onsetflag = maxlocation - 10  # -10 is from experience
                else:
                    onsetflag = -1

        timedelay = (potential[onsetflag] - standard) / sr
        arrive = flag[k - 1] + timedelay
        if onsetflag < 0:
            arrive = -1
        on.append(arrive)

        # plt.plot(kur/max(kur)+k)
        '''
        if onsetflag > 0:
            plt.plot(a / max(abs(a)) + 2 * k, linewidth=0.7, marker='|', ls='-', markeredgecolor='black', ms=5,
                     markevery=[potential[onsetflag]])
        else:
            plt.plot(a / max(abs(a)) + 2 * k,linewidth=0.7, ls='-')
        '''

    #plt.show()
    #fff.set_aspect(30)
    #plt.savefig(dirsave + str(i) + 's2.eps')

    s2onsets.append(on)

s2onsets=np.array(s2onsets)
print(s2onsets)
pickle.dump([s2onsets],open(dirsave + 's2onsets.p','wb'))








