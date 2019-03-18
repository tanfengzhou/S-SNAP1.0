import numpy as np
import pickle
import taupz

with open('./tableP.p' , 'rb') as f:
    comein=pickle.load(f)

tableP = comein[0]

with open('./tableS.p' , 'rb') as f:
    comein=pickle.load(f)

tableS = comein[0]

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

studyarea=[0, 0, 9.5, 9.5]
studydepth=[800,6000]
ygrid=0.25     # km
xgrid=0.25     # km
depgrid=250    # meter
ys=np.arange(studyarea[0],studyarea[0]+studyarea[2],ygrid)
xs=np.arange(studyarea[1],studyarea[1]+studyarea[3],xgrid)
deps=np.arange(studydepth[0],studydepth[1]+depgrid,depgrid)
studygrids=[]
for i in ys:
    for j in xs:
        for k in deps:
            studygrids.append([i,j,k])

print(len(studygrids))
pickle.dump([studygrids],open('studygrids.p','wb'))

import distance
traveldis=[]
for i in studygrids:
    a=[]
    for j,k in zip(sty,stx):
        a.append(( (j-i[0])**2 + (k-i[1])**2 ) ** 0.5 / 111)

    traveldis.append(a)

ptraveltimes=[]
straveltimes=[]
for i in range(0,len(traveldis)):
    if i%1000==0:
        print(i)
    a=[]
    b=[]
    for j in range(0, len(traveldis[i])):
        timeP = taupz.taupz(tableP, tableS, studygrids[i][2]/1000, traveldis[i][j],'p', stz[j]-900)
        a.append( timeP )
        timeS = taupz.taupz(tableP, tableS, studygrids[i][2]/1000, traveldis[i][j],'s', stz[j]-900)
        b.append( timeS )

    ptraveltimes.append(a)
    straveltimes.append(b)

pickle.dump([ptraveltimes],open('ptraveltimes_table.p','wb'))
pickle.dump([straveltimes],open('straveltimes_table.p','wb'))

