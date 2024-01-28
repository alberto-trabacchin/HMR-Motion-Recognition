from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_dataset(walking_ds, squatting_ds, var="x"):
    labelsize = 14
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(walking_ds[var], label="walking")
    ax.plot(squatting_ds[var], label="squatting")
    ax.set_xlabel(f"$time[s]$", fontsize=labelsize)
    ax.set_ylabel(f"${var}-acceleration [m/s^2]$", fontsize=labelsize)
    ax.legend()
    ax.grid()
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