import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot(u, timesteps, dt, savefile=None):
    plate_length = int(np.sqrt(u.shape[1]))
    u = u.reshape((timesteps, plate_length, plate_length))

    def plotheatmap(u_k, k):
        # Clear the current plot figure
        plt.clf()
        plt.title(f"Temperature at t = {k * dt:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")

        # This is to plot u_k (u at time-step k)
        plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar()

        return plt

    def animate(k):
        plotheatmap(u[k], k)

    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=timesteps, repeat=False)
    if savefile:
        anim.save(savefile)

    return anim
