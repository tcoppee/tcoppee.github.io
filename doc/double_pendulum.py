import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Circle
import os
from PIL import Image

def deriv(y, t, L, m):
    # y = theta, z1, alpha, z2
    theta, z1, alpha, z2 = y
    z1dot = (z1*z1*np.sin(alpha)*(1+np.cos(alpha)) + 2*z1*z2*np.sin(alpha) + z2*z2*np.sin(alpha) + (g/L)*(np.sin(theta+alpha)*np.cos(alpha)-2*np.sin(theta)))/(2-np.cos(alpha)*np.cos(alpha))
    z2dot = -(g/L)*np.sin(theta+alpha)-z1dot*(1+np.cos(alpha))-z1*z1*np.sin(alpha)
    thetadot = z1
    alphadot = z2
    
    return thetadot, z1dot, alphadot, z2dot

L = 10 # Rod lengths (m)
m = 1 # Masses (kg)
compute = True

g = 9.81 # Gravitational acceleration (ms^2)
r=1

tmax = 200 # (s)
dt = 0.01 # (s)
t = np.arange(0,tmax+dt,dt)

# Initial conditions: theta, thetadot, alpha, alphadot
y0 = np.array([np.pi,0,np.pi*0.1,0])

# Numerical integration
y = odeint(deriv, y0, t, args=(L,m))
theta = y[:,0]
alpha = y[:,2]

x1 = L*np.sin(theta)
y1 = -L*np.cos(theta)
x2 = x1+L*np.sin(theta+alpha)
y2 = y1-L*np.cos(theta+alpha)

def make_plot(i):
    ax.plot([0,x1[i],x2[i]],[0,y1[i],y2[i]], lw=3,c='k')
    c0 = Circle((0,0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i],y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i],y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    
    ax.set_xlim(-2*L-r, 2*L+r)
    ax.set_ylim(-2*L-r, 2*L+r)
    ax.set_aspect('equal',adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
    plt.cla()

if compute:
    fps = 10
    di = int(1/fps/dt)
    fig = plt.figure(dpi=72)
    ax = fig.add_subplot(111)
    for i in range(0, t.size):
        make_plot(i)
    
frames_folder = 'frames'
frames = sorted([os.path.join(frames_folder,f) for f in os.listdir(frames_folder) if f.endswith(('.png'))])

images = [Image.open(frame) for frame in frames]

output_gif = "output.gif"
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=100,  # Durée par frame en ms
    loop=0  # 0 pour une boucle infinie
)

