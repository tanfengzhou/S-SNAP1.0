def beta(dep):
    if dep>6000:
        dep=6000
    import scipy.io as sio
    v=sio.loadmat('velocity')
    b=v['vs']/1000
    return(b[int(dep)][0])
