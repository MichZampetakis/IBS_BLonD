import numpy as np

def prepareTwiss(twissFile): 
    """
    Prepares the data from a MADX Twiss file including at least {s, l, betx, bety, dx, dpx, dy, dpy}
    Inputs : twissFile : [str] twiss file (default=None)
    Returns: Dictionary
    """
    if twissFile is None:
        raise IOError('# loadlattice::prepareTwiss: You need to define Madx twiss file in [prepareData]')
    with open(twissFile, 'r') as f:
        for line in enumerate(f.readlines()):
            if line[1][0] == '*':
                skip_header_nr = line[0]
            elif line[1][0] == '$':
                skip_rows_nr = line[0]+1
                break

    header = np.genfromtxt(twissFile, skip_header = skip_header_nr, max_rows = 1, dtype = str)
    data = np.loadtxt(twissFile, skiprows = skip_rows_nr, 
                      usecols = (np.where(header == 'S')[0][0] - 1, np.where(header == 'L')[0][0] - 1,
                                 np.where(header == 'BETX')[0][0] - 1, np.where(header == 'BETY')[0][0] - 1,
                                 np.where(header == 'ALFX')[0][0] - 1, np.where(header == 'ALFY')[0][0] - 1,
                                 np.where(header == 'DX')[0][0] - 1, np.where(header == 'DPX')[0][0] - 1,
                                 np.where(header == 'DY')[0][0] - 1, np.where(header == 'DPY')[0][0] - 1))

    twiss = {'position': data[:,0], 'length': data[:,1], 'betx': data[:,2], 'bety': data[:,3], 
             'alfx': data[:,4], 'alfy': data[:,5], 'dx': data[:,6], 'dpx': data[:,7], 
             'dy': data[:,8], 'dpy': data[:,9]}

    return twiss