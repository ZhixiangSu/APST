import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


data=[]
with open("ablation_study.csv",'r') as f:
    for line in f:
        d=line.split(',')
        d=[float(di) for di in d]
        data.append(d)

for d in data:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    z = np.array([[d[0],d[1]],[d[2],d[3]]])
    offset=float(int(np.min(z)*10-1)/10)
    z=z-offset
    xedges=np.arange(3)
    yedges=np.arange(3)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = z.ravel()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks(np.arange(0,1,0.05))
    ax.set_zticklabels(np.around(np.arange(0,1,0.05)+offset,1))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',color=['red','green','yellow','blue'],alpha=0.8)
    # plt.ylim(0.4,1)
    # plt.bar(np.arange(4),d,color=['red','green','yellow','blue'])

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='yellow', lw=4),
                    Line2D([0],[0],color='blue',lw=4)]
    plt.legend(custom_lines,['KRST (short description)','KRST (long description)','APST (short description)','APST (long description)'],loc='lower right')
    plt.show()