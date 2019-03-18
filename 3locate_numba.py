import pickle
import numpy as np
import distance
import taupz
from numba import jit

@jit(nopython=True)
def maxiscan(pstation, parrival, ptraveltimes, terr, maxi):
    for j in range(0, len(pstation) - 1):
        for k in range(j + 1, len(pstation)):
            obdiff = parrival[j] - parrival[k]
            for l in range(0, len(ptraveltimes)):
                thdiff = ptraveltimes[l][pstation[j]] - ptraveltimes[l][pstation[k]]
                if abs(thdiff - obdiff) < terr:
                    maxi[l] = maxi[l] + 1

    return (maxi)


dirread='./H3/'
dirsave='./H3/'

sr=500
win=1
step=0.5
highq=15
lowq=4
terr=0.1          # acceptable error in second
Q_threshold=0.5
outlier = 0.5  # second

import getbeta

with open(dirread + 'ponsets.p' , 'rb') as f:
    comein=pickle.load(f)

ponsets=comein[0]

with open(dirread + 's1onsets.p' , 'rb') as f:
    comein=pickle.load(f)

s1onsets=comein[0]

with open(dirread + 's2onsets.p' , 'rb') as f:
    comein=pickle.load(f)

s2onsets=comein[0]


with open('./ptraveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)

ptraveltimes = comein[0]

with open('./straveltimes_table.p' , 'rb') as f:
    comein=pickle.load(f)

straveltimes = comein[0]

with open('./studygrids.p', 'rb') as  f:
    comein = pickle.load(f)

studygrids = comein[0]

with open('./tableP.p' , 'rb') as f:
    comein=pickle.load(f)

tableP = comein[0]

with open('./tableS.p' , 'rb') as f:
    comein=pickle.load(f)

tableS = comein[0]

stz=[]
sty=[]
stx=[]
readin = open('./stz','r')
lines = readin.readlines()
for i in lines:
    stz.append(float(i))

readin = open('./sty','r')
lines = readin.readlines()
for i in lines:
    sty.append(float(i))

readin = open('./stx','r')
lines = readin.readlines()
for i in lines:
    stx.append(float(i))



total=len(ponsets)
print('There are '+str(total)+' events.\n')
events=[]
MAXI=[]
catalog=[]

for i in range(0,total):

    startrefinelatgrid = 0.25
    startrefinelongrid = 0.25
    startrefinedepgrid = 250

    p=ponsets[i]
    s1=s1onsets[i]
    s2=s2onsets[i]

    positivep=0
    parrival=[]
    pstation=[]
    num=0
    for j in p:
        if j>0:
            parrival.append(j)
            pstation.append(num)
            positivep = positivep + 1
        num=num+1

    positives = 0
    sarrival = []
    sstation = []
    num = 0
    for j, k in zip(s1, s2):
        if j > 0 or k > 0:
            positives = positives + 1
            sstation.append(num)
            if j > 0 and k > 0:
                sarrival.append((j + k) / 2)
            else:
                if j > 0:
                    sarrival.append(j)
                if k > 0:
                    sarrival.append(k)
        num = num + 1

    if positivep < lowq and positives < lowq:
        print('No.'+str(i)+' error: not enough picking. \n')
        events.append([-1,-1,-1,-1,-1,-1])
        MAXI.append(np.zeros(len(ptraveltimes)))
        continue

    fullp = positivep
    fulls = positives
    MAXI_ceil = fullp*(fullp-1)/2 + fulls*(fulls-1)/2

    maxi=np.zeros(len(ptraveltimes))
    if positivep >= lowq:
        maxi=maxiscan(pstation, parrival, np.array(ptraveltimes), terr, maxi)

    if positives >= lowq:
        maxi=maxiscan(sstation, sarrival, np.array(straveltimes), terr, maxi)

    m = max(maxi)
    Q = m/MAXI_ceil

    if positivep < highq or positives < highq:
        Q = 0

    print('No.' + str(i) + ' : Quality = ' + str(Q))

    p = [j for j, k in enumerate(maxi) if k == m]

    evla = studygrids[p[int(np.floor(len(p)/2))]][0]
    evlo = studygrids[p[int(np.floor(len(p)/2))]][1]
    evdep = studygrids[p[int(np.floor(len(p)/2))]][2]
    p[0] = p[int(np.floor(len(p)/2))]

    estimatetimes=[]
    for j in range(0,len(pstation)):
        t=parrival[j]-ptraveltimes[p[0]][pstation[j]]
        estimatetimes.append(t)

    for j in range(0,len(sstation)):
        t=sarrival[j]-straveltimes[p[0]][sstation[j]]
        estimatetimes.append(t)

    eventtime0=np.median(estimatetimes)

    events.append([Q, m, eventtime0, evla, evlo, evdep])
    MAXI.append(maxi)

    if Q < Q_threshold:
        print('No.' + str(i) + ' low quality event \n')
        catalog.append([eventtime0, evla, evlo, evdep, -1, -1])
        print([eventtime0, evla, evlo, evdep, -1, -1])
        continue

    startrefinelatgrid = startrefinelatgrid/2  # degree
    startrefinelongrid = startrefinelongrid/2  # degree
    startrefinedepgrid = startrefinedepgrid/2  # meter
    ys = np.arange(evla-startrefinelatgrid*2, evla+startrefinelatgrid*2+startrefinelatgrid/10, startrefinelatgrid)
    xs = np.arange(evlo-startrefinelongrid*2, evlo+startrefinelongrid*2+startrefinelongrid/10, startrefinelongrid)
    deps = np.arange(evdep-startrefinedepgrid*2, evdep+startrefinedepgrid*2+startrefinedepgrid/10, startrefinedepgrid)
    refinestudygrids = []
    for l in ys:
        for j in xs:
            for k in deps:
                refinestudygrids.append([l, j, k])

    refinetraveldis = []
    for l in refinestudygrids:
        a = []
        for j, k in zip(sty, stx):
            a.append(( (j-l[0])**2 + (k-l[1])**2 ) ** 0.5 / 111)

        refinetraveldis.append(a)

    beta = getbeta.beta(evdep)
    timerangep = np.arange(0, startrefinedepgrid * 2 * 1.73 / 1000 / beta + startrefinedepgrid * 2  * 1.73 / 1000 / beta / 10, startrefinedepgrid /1000/ beta )
    timerangen = -np.flip(timerangep,0)

    timerange = []
    for k in timerangen:
        timerange.append(k)
    timerange.pop()
    for k in timerangep:
        timerange.append(k)

    refineptraveltimes = []
    refinestraveltimes = []
    for l in range(0, len(refinetraveldis)):
        a = []
        for j in range(0, len(refinetraveldis[l])):
            time=taupz.taupz(tableP, tableS, refinestudygrids[l][2] / 1000, refinetraveldis[l][j], 'p', stz[j]-900)
            a.append(time)

        refineptraveltimes.append(a)

        b = []
        for j in range(0, len(refinetraveldis[l])):
            time=taupz.taupz(tableP, tableS, refinestudygrids[l][2] / 1000, refinetraveldis[l][j], 's', stz[j]-900)
            b.append(time)

        refinestraveltimes.append(b)

    for j in range(0,len(pstation)):
        ot=parrival[j]-eventtime0
        tht=ptraveltimes[p[0]][pstation[j]]
        if abs(ot-tht) > outlier:
            pstation[j]=-1

    for j in range(0,len(sstation)):
        ot=sarrival[j]-eventtime0
        tht=straveltimes[p[0]][sstation[j]]
        if abs(ot-tht) > outlier:
            sstation[j]=-1

    remains=[]
    potimes=[]
    for j in range(0,len(refinestudygrids)):
        minremain = 9999
        n=-1
        #remain=0

        for k in timerange:
        #for k in np.arange(-0.12,0.121,0.04):
            remain = 0
            potime=eventtime0+k
            n=n+1
            clean=0
            for l in range(0,len(pstation)):
                if pstation[l] > 0:
                    clean=clean+1
                    remain=remain+(refineptraveltimes[j][pstation[l]]-(parrival[l]-potime))**2
            for l in range(0,len(sstation)):
                if sstation[l] > 0:
                    clean = clean + 1
                    remain=remain+(refinestraveltimes[j][sstation[l]]-(sarrival[l]-potime))**2

            remain=(remain/clean)**0.5
            if remain<minremain:
                minremain=remain
                minrenum=n

        remains.append(minremain)
        potimes.append(eventtime0+timerange[minrenum])

    minimum=min(remains)
    print('minimum = '+str(minimum)+'s')
    print('clean stations: ' + str(clean))
    p = [j for j, k in enumerate(remains) if k == minimum]
    finaltime=potimes[p[0]]
    finallat, finallon, finaldep = refinestudygrids[p[0]][0], refinestudygrids[p[0]][1], refinestudygrids[p[0]][2]

########################################################################################################################
    refinelatgrid = startrefinelatgrid
    refinelongrid = startrefinelongrid
    refinedepgrid = startrefinedepgrid

    while refinedepgrid > 1:

        evla, evlo, evdep, eventtime0 = finallat, finallon, finaldep, finaltime
        refinelatgrid = refinelatgrid/2  # degree
        refinelongrid = refinelongrid/2  # degree
        refinedepgrid = refinedepgrid/2  # meter
        ys = np.arange(evla - refinelatgrid, evla + refinelatgrid+refinelatgrid/10, refinelatgrid)
        xs = np.arange(evlo - refinelongrid, evlo + refinelongrid+refinelongrid/10, refinelongrid)
        deps = np.arange(evdep - refinedepgrid, evdep + refinedepgrid+refinedepgrid/10, refinedepgrid)
        refinestudygrids = []
        for l in ys:
            for j in xs:
                for k in deps:
                    refinestudygrids.append([l, j, k])

        refinetraveldis = []
        for l in refinestudygrids:
            a = []
            for j, k in zip(sty, stx):
                a.append(( (j-l[0])**2 + (k-l[1])**2 ) ** 0.5 / 111)

            refinetraveldis.append(a)

        beta = getbeta.beta(evdep)
        timerangep = np.arange(0,refinedepgrid * 1.73 / 1000 / beta + refinedepgrid * 1.73 / 1000 / beta / 10, refinedepgrid / 1000 / beta)
        timerangen = -np.flip(timerangep,0)

        timerange = []
        for k in timerangen:
            timerange.append(k)
        timerange.pop()
        for k in timerangep:
            timerange.append(k)

        refineptraveltimes = []
        refinestraveltimes = []
        for l in range(0, len(refinetraveldis)):
            a = []
            for j in range(0, len(refinetraveldis[l])):
                time = taupz.taupz(tableP, tableS, refinestudygrids[l][2] / 1000, refinetraveldis[l][j], 'p', stz[j]-900)
                a.append(time)

            refineptraveltimes.append(a)

            b = []
            for j in range(0, len(refinetraveldis[l])):
                time = taupz.taupz(tableP, tableS, refinestudygrids[l][2] / 1000, refinetraveldis[l][j], 's', stz[j]-900)
                b.append(time)

            refinestraveltimes.append(b)


        remains = []
        potimes = []
        for j in range(0, len(refinestudygrids)):
            minremain = 9999
            n = -1
            for k in timerange:
                remain=0
                potime = eventtime0 + k
                n = n + 1
                clean = 0
                for l in range(0, len(pstation)):
                    if pstation[l] > 0:
                        clean = clean + 1
                        remain = remain + (refineptraveltimes[j][pstation[l]] - (parrival[l] - potime)) ** 2
                for l in range(0, len(sstation)):
                    if sstation[l] > 0:
                        clean = clean + 1
                        remain = remain + (refinestraveltimes[j][sstation[l]] - (sarrival[l] - potime)) ** 2

                remain = (remain / clean) ** 0.5
                if remain < minremain:
                    minremain = remain
                    minrenum = n

            remains.append(minremain)
            potimes.append(eventtime0 + timerange[minrenum])

        improve = (minimum-min(remains))/minimum
        minimum = min(remains)
        print('refinegrid = '+str(refinedepgrid)+'m')
        print('minimum = '+str(minimum)+'s')
        print('improve = '+str(improve*100)+"%")
        print('clean stations: ' + str(clean))
        p = [j for j, k in enumerate(remains) if k == minimum]
        finaltime = potimes[p[0]]
        finallat, finallon, finaldep = refinestudygrids[p[0]][0], refinestudygrids[p[0]][1], refinestudygrids[p[0]][2]
        finalgrid = refinedepgrid
        if improve < 0.001:
            refinelatgrid = startrefinelatgrid
            refinelongrid = startrefinelongrid
            refinedepgrid = startrefinedepgrid
            break

        elif refinedepgrid < 1:
            refinelatgrid = startrefinelatgrid
            refinelongrid = startrefinelongrid
            refinedepgrid = startrefinedepgrid
            break

########################################################################################################################

    catalog.append([finaltime, finallat, finallon, finaldep, minimum, finalgrid, clean])
    print([finaltime, finallat, finallon, finaldep, minimum, finalgrid, clean])

pickle.dump([events],open(dirsave + 'MAXI_PRED.p','wb'))
pickle.dump([MAXI],open(dirsave + 'MAXIvalue.p','wb'))
pickle.dump([catalog],open(dirsave + 'cataloghigh.p','wb'))











