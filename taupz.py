def taupz(tableP, tableS, dep, dis, phase, al, depgrid=0.01, disgrid=0.0001):

    # dep in km, al in m.

    import math

    if phase=='p':
        table=tableP
        v1=2.5
    if phase=='s':
        table=tableS
        v1=1.25

    if dep < 0.8:
        dep = 0.8

    dep = dep - 0.8

    if dis < disgrid:
        dis=disgrid

    time1=table[int(math.floor(dis/disgrid))][int(math.floor(dep/depgrid))]
    time2=table[int(math.floor(dis/disgrid))][int(math.ceil(dep/depgrid))]
    time3=table[int(math.ceil(dis/disgrid))][int(math.ceil(dep/depgrid))]

    time=time1 + (dis/disgrid-math.floor(dis/disgrid))*(time3-time1) + (dep/depgrid-math.floor(dep/depgrid))*(time2-time1)

    time = time + al / 1000 / v1

    return(time)