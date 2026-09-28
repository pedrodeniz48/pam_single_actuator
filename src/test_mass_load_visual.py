from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments_real_time import run_experiment
import activation_funcs as act


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    ROOT
    / "results"
    / "mass_load_test"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment configuration
# ============================================================

F_MAX = 1650.0

MASS_MIN = 1.0
MASS_MAX = 30.0
MASS_STEP = 2.0

DURATION = 12.0

# Ignore the initial transient when computing the metrics
ANALYSIS_START_TIME = 4.0


# ============================================================
# Mass values
# ============================================================

masses = np.arange(
    MASS_MIN,
    MASS_MAX + MASS_STEP,
    MASS_STEP
)


# ============================================================
# Run experiments
# ============================================================

summary = []

all_runs = {}


for mass in masses:

    print()
    print("=" * 60)
    print(
        f"Running experiment:"
        f" mass = {mass:.1f} kg,"
        f" f_max = {F_MAX:.1f} N"
    )
    print("=" * 60)

    df = run_experiment(
        duration=DURATION,
        mass=mass,
        activation_function=act.step,
        muscle_parameters={
            "f_max": F_MAX,
        },
        show_viewer=True,
    )

    # Store in memory for later plotting
    all_runs[mass] = df

    # --------------------------------------------------------
    # Save complete time series
    # --------------------------------------------------------

    output_file = (
        RESULTS_PATH
        / f"mass_{mass:.1f}kg.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Steady-state analysis
    # --------------------------------------------------------

    steady = df[
        df["time"] >= ANALYSIS_START_TIME
    ]

    # --------------------------------------------------------
    # Position / contraction
    # --------------------------------------------------------

    max_contraction = (
        steady["tendon_contraction"].max()
    )

    min_contraction = (
        steady["tendon_contraction"].min()
    )

    contraction_amplitude = (
        max_contraction
        - min_contraction
    )

    mean_contraction = (
        steady["tendon_contraction"].mean()
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    max_velocity = (
        steady["tendon_velocity"]
        .abs()
        .max()
    )

    mean_abs_velocity = (
        steady["tendon_velocity"]
        .abs()
        .mean()
    )

    rms_velocity = np.sqrt(
        np.mean(
            steady["tendon_velocity"] ** 2
        )
    )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary.append(
        {
            "mass": mass,
            "f_max": F_MAX,

            "max_contraction":
                max_contraction,

            "min_contraction":
                min_contraction,

            "mean_contraction":
                mean_contraction,

            "contraction_amplitude":
                contraction_amplitude,

            "max_tendon_velocity":
                max_velocity,

            "mean_abs_tendon_velocity":
                mean_abs_velocity,

            "rms_tendon_velocity":
                rms_velocity,
        }
    )


# ============================================================
# Create summary DataFrame
# ============================================================

summary_df = pd.DataFrame(summary)


summary_file = (
    RESULTS_PATH
    / "mass_load_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


print()
print("All simulations completed.")
print(f"Summary saved to: {summary_file}")


# ============================================================
# GRAPH 1
# Mass vs contraction amplitude
# ============================================================

fig1 = plt.figure()

plt.plot(
    summary_df["mass"],
    summary_df["contraction_amplitude"],
    marker="o",
)

plt.xlabel("Mass [kg]")
plt.ylabel(
    "Tendon contraction amplitude [m]"
)

plt.title(
    "Mass vs Tendon Contraction"
)

plt.grid(True)
plt.tight_layout()

fig1.savefig(
    RESULTS_PATH
    / "mass_vs_contraction.png",
    dpi=300,
)


# ============================================================
# GRAPH 2
# Mass vs RMS tendon velocity
# ============================================================

fig2 = plt.figure()

plt.plot(
    summary_df["mass"],
    summary_df["rms_tendon_velocity"],
    marker="o",
)

plt.xlabel("Mass [kg]")
plt.ylabel(
    "RMS Tendon Velocity [m/s]"
)

plt.title(
    "Mass vs Tendon Velocity"
)

plt.grid(True)
plt.tight_layout()

fig2.savefig(
    RESULTS_PATH
    / "mass_vs_velocity.png",
    dpi=300,
)


# ============================================================
# GRAPH 3
# Contraction over time for every mass
# ============================================================

fig3 = plt.figure()

for mass, df in all_runs.items():

    plt.plot(
        df["time"],
        df["tendon_contraction"],
        label=f"{mass:.0f} kg",
    )

plt.xlabel("Time [s]")
plt.ylabel(
    "Tendon Contraction [m]"
)

plt.title(
    "Tendon Contraction Over Time"
)

plt.grid(True)

plt.legend(
    title="Mass",
    fontsize="small",
)

plt.tight_layout()

fig3.savefig(
    RESULTS_PATH
    / "contraction_time_series.png",
    dpi=300,
)


# ============================================================
# GRAPH 4
# Tendon velocity over time
# ============================================================

fig4 = plt.figure()

for mass, df in all_runs.items():

    plt.plot(
        df["time"],
        df["tendon_velocity"],
        label=f"{mass:.0f} kg",
    )

plt.xlabel("Time [s]")
plt.ylabel(
    "Tendon Velocity [m/s]"
)

plt.title(
    "Tendon Velocity Over Time"
)

plt.grid(True)

plt.legend(
    title="Mass",
    fontsize="small",
)

plt.tight_layout()

fig4.savefig(
    RESULTS_PATH
    / "velocity_time_series.png",
    dpi=300,
)


# ============================================================
# Show ALL figures simultaneously
# ============================================================

plt.show()