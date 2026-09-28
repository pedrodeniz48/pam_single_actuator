from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments import run_experiment
import activation_funcs as act
from report_utils import document_experiment


ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = ROOT / "results" / "mass_load_test"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment configuration
# ============================================================

F_MAX = 1500

MASS_MIN = 11
MASS_MAX = 15
MASS_STEP = 0.25

DURATION = 5.0

# Step happens at t = 1 s
ANALYSIS_START_TIME = 1.0


# ============================================================
# Mass range
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

# Keep full runs for diagnostic plots
all_runs = {}


for mass in masses:

    print(
        f"Running mass = {mass:.1f} kg "
        f"with f_max = {F_MAX:.1f} N"
    )

    df = run_experiment(
        duration=DURATION,
        mass=mass,
        activation_function=act.step,
        muscle_parameters={
            "f_max": F_MAX,
        }
    )

    all_runs[mass] = df

    # ========================================================
    # Save complete time series
    # ========================================================

    output_file = (
        RESULTS_PATH
        / f"mass_{mass:.1f}kg.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # Analyze response after activation step
    # ========================================================

    response = df[
        df["time"] >= ANALYSIS_START_TIME
    ].copy()

    initial_response_length = (
        response["tendon_length"].iloc[0]
    )

    response["step_contraction"] = (
        initial_response_length
        - response["tendon_length"]
    )

    max_contraction = (
        response["step_contraction"].max()
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    max_contraction = (
        response["tendon_contraction"].max()
    )

    max_tendon_velocity = (
        response["tendon_velocity"]
        .abs()
        .max()
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.append(
        {
            "mass": mass,
            "f_max": F_MAX,
            "max_contraction": max_contraction,
            "max_tendon_velocity": max_tendon_velocity,
        }
    )


# ============================================================
# Save summary
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

print(
    f"\nSummary saved to:\n{summary_file}"
)

# ============================================================
# Find critical mass
# ============================================================

# First derivative:
# how much contraction changes per kg
summary_df["contraction_slope"] = np.gradient(
    summary_df["max_contraction"],
    summary_df["mass"]
)

# Second derivative:
# how much the slope itself changes
summary_df["contraction_slope_change"] = np.gradient(
    summary_df["contraction_slope"],
    summary_df["mass"]
)

# First derivative:
# how much velocity changes per kg
summary_df["velocity_slope"] = np.gradient(
    summary_df["max_tendon_velocity"],
    summary_df["mass"]
)

# Second derivative:
# how much the slope itself changes
summary_df["velocity_slope_change"] = np.gradient(
    summary_df["velocity_slope"],
    summary_df["mass"]
)

# Avoid using boundary points because numerical
# derivatives are less reliable there
interior = summary_df.iloc[1:-1]

critical_mass_index = (
    interior["contraction_slope_change"]
    .abs()
    .idxmax()
)

critical_mass = summary_df.loc[
    critical_mass_index,
    "mass"
]

critical_vel_index = (
    interior["velocity_slope_change"]
    .abs()
    .idxmax()
)

critical_vel = summary_df.loc[
    critical_vel_index,
    "mass"
]

print(
    f"\nApproximate critical mass per contraction analysis: "
    f"{critical_mass:.2f} kg"
)

print(
    f"\nApproximate critical mass per velocity analysis: "
    f"{critical_vel:.2f} kg"
)

# ============================================================
# GRAPH 1
# Mass vs maximum tendon contraction
# ============================================================

fig1 = plt.figure()

plt.plot(
    summary_df["mass"],
    summary_df["max_contraction"],
    marker="o",
    markersize=4,
)

plt.xlabel("Mass [kg]")
plt.ylabel(
    "Maximum Tendon Contraction [m]"
)

plt.title(
    "Effect of Load Mass on Maximum PAM Contraction"
)

plt.grid(True)
plt.tight_layout()

fig1.savefig(
    RESULTS_PATH
    / "mass_vs_max_contraction.png",
    dpi=300,
)


# ============================================================
# GRAPH 2
# Mass vs maximum tendon velocity
# ============================================================

fig2 = plt.figure()

plt.plot(
    summary_df["mass"],
    summary_df["max_tendon_velocity"],
    marker="o",
    markersize=4,
)

plt.xlabel("Mass [kg]")
plt.ylabel(
    "Maximum Absolute Tendon Velocity [m/s]"
)

plt.title(
    "Effect of Load Mass on Maximum Tendon Velocity"
)

plt.grid(True)
plt.tight_layout()

fig2.savefig(
    RESULTS_PATH
    / "mass_vs_max_velocity.png",
    dpi=300,
)


# ============================================================
# GRAPH 3
# Tendon contraction over time
# Diagnostic plot
# ============================================================

fig3 = plt.figure()

for mass, df in all_runs.items():

    plt.plot(
        df["time"],
        df["tendon_contraction"],
        label=f"{mass:.0f} kg",
    )

plt.axvline(
    x=ANALYSIS_START_TIME,
    linestyle="--",
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
# Diagnostic plot
# ============================================================

fig4 = plt.figure()

for mass, df in all_runs.items():

    plt.plot(
        df["time"],
        df["tendon_velocity"],
        label=f"{mass:.0f} kg",
    )

plt.axvline(
    x=ANALYSIS_START_TIME,
    linestyle="--",
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
# Generate report
# ============================================================

parameters = {
    "f_max [N]": F_MAX,
    "mass_min [kg]": MASS_MIN,
    "mass_max [kg]": MASS_MAX,
    "mass_step [kg]": MASS_STEP,
    "duration [s]": DURATION,
    "analysis_start_time [s]": ANALYSIS_START_TIME,
    "activation_function": "step",
}

key_results = {
    "Critical mass - contraction [kg]":
        critical_mass,

    "Critical mass - velocity [kg]":
        critical_vel,
}

figure_paths = [
    RESULTS_PATH
    / "mass_vs_max_contraction.png",

    RESULTS_PATH
    / "mass_vs_max_velocity.png",

    RESULTS_PATH
    / "contraction_time_series.png",

    RESULTS_PATH
    / "velocity_time_series.png",
]

document_experiment(
    title="PAM Mass Load Test",

    description=(
        "Evaluation of the effect of load mass on "
        "maximum tendon contraction and maximum tendon "
        "velocity while maintaining a fixed maximum "
        "contractile force."
    ),

    parameters=parameters,

    summary_df=summary_df,

    key_results=key_results,

    figure_paths=figure_paths,

    results_directory=RESULTS_PATH,

    max_rows=20,

    filename_prefix="mass_load_report",
)

# ============================================================
# Show all graphs simultaneously
# ============================================================

plt.show()