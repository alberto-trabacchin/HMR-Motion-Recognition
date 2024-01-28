from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_dataset(walking_ds, squatting_ds, var="x"):
    labelsize = 12
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(walking_ds[var], label="walking")
    ax[0].plot(squatting_ds[var], label="squatting")
    ax[0].set_xlabel(f"$time[s]$", fontsize=labelsize)
    ax[0].set_ylabel(f"${var}-acceleration [m/s^2]$", fontsize=labelsize)
    ax[0].legend()
    ax[1].plot(walking_ds[var], label="walking")
    ax[1].plot(squatting_ds[var], label="squatting")
    ax[1].set_xlabel(f"$time[s]$", fontsize=labelsize)
    ax[1].set_ylabel(f"${var}-acceleration [m/s^2]$", fontsize=labelsize)
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    return fig

if __name__ == "__main__":
    walking_ds = pd.read_csv("data/walking/Accelerometer.csv", parse_dates=True)
    squatting_ds = pd.read_csv("data/squatting/Accelerometer.csv", parse_dates=True)
    x_fig = plot_dataset(walking_ds, squatting_ds, var="x")
    y_fig = plot_dataset(walking_ds, squatting_ds, var="y")
    z_fig = plot_dataset(walking_ds, squatting_ds, var="z")
    Path("./plots").mkdir(parents=True, exist_ok=True)
    x_fig.savefig("./plots/x_acceleration.png", dpi=600)
    y_fig.savefig("./plots/y_acceleration.png", dpi=600)
    z_fig.savefig("./plots/z_acceleration.png" ,dpi=600)