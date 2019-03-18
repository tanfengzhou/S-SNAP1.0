import pickle
import numpy as np

stz=[]
sty=[]
stx=[]

readin = open('stz','r')
lines = readin.readlines()
for i in lines:
    stz.append(float(i))

readin = open('stx','r')
lines = readin.readlines()
for i in lines:
    stx.append(float(i))

readin = open('sty','r')
lines = readin.readlines()
for i in lines:
    sty.append(float(i))

readdir = './H3/'
savedir = readdir

with open(readdir + 'MAXI_PRED.p' , 'rb') as f:
    comein=pickle.load(f)

events = comein[0]

with open(readdir + 'magnitudeauto.p', 'rb') as f:
    comein=pickle.load(f)

magnitude = comein[0]


f= open(savedir + 'magnitudeauto.txt', 'w')
for i in magnitude:
    for j in i:
        f.write(str(j))
        f.write(' ')

    f.write('\n')

f.close()


f= open(savedir + 'events.txt', 'w')
for i in events:
    for j in i:
        f.write(str(j))
        f.write(' ')

    f.write('\n')

f.close()

evlas=[]
evlos=[]
evdeps=[]


for i in events:
    evlas.append(i[3])
    evlos.append(i[4])
    evdeps.append(i[5])

with open(readdir + 'cataloghigh.p' , 'rb') as f:
    comein=pickle.load(f)

catalog=comein[0]

f= open(savedir + 'cataloghigh.txt', 'w')
for i in catalog:
    for j in i:
        f.write(str(j))
        f.write(' ')

    f.write('\n')

f.close()

evlashigh=[]
evloshigh=[]
evdepshigh=[]

evlaslow=[]
evloslow=[]
evdepslow=[]

for i in magnitude:
    if i[5] > 0:
        evlashigh.append(i[1])
        evloshigh.append(i[2])
        evdepshigh.append(i[3])
    else:
        evlaslow.append(i[1])
        evloslow.append(i[2])
        evdepslow.append(i[3])

import stations_plot
studyarea=[0, 0, 9.5, 9.5]
stations_plot.p(lat0=studyarea[0],lat=studyarea[2],lon0=studyarea[1],lon=studyarea[3],stlas=evlashigh,stlos=evloshigh,name= savedir +'cataloghigh.pdf',marker='.',color='r')
stations_plot.p(lat0=studyarea[0],lat=studyarea[2],lon0=studyarea[1],lon=studyarea[3],stlas=evlaslow,stlos=evloslow,name= savedir +'cataloglow.pdf',marker='.',color='b')
stations_plot.p(lat0=studyarea[0],lat=studyarea[2],lon0=studyarea[1],lon=studyarea[3],stlas=sty,stlos=stx,name= savedir +'stations.pdf',marker='^',color='g')

print(0)

