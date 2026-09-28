from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments import run_experiment
from report_utils import document_experiment


ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    ROOT
    / "results"
    / "frequency_velocity_test"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment configuration
# ============================================================

MASS = 1.0

F_MAX = 1500.0

FREQUENCIES = np.arange(0.8, 2, 0.05)

NUMBER_OF_CYCLES = 5
IGNORE_CYCLES = 2


# ============================================================
# Experiment
# ============================================================

summary = []
all_runs = {}


for frequency in FREQUENCIES:

    print(
        f"Running frequency = {frequency:.2f} Hz"
    )

    # --------------------------------------------------------
    # Activation function for current frequency
    # --------------------------------------------------------

    def activation_sine(t, f=frequency):

        return (
            0.055
            - 0.045
            * math.cos(
                2 * math.pi * f * t
            )
        )

    # One period
    period = 1.0 / frequency

    # Total experiment duration
    duration = NUMBER_OF_CYCLES * period

    # Time after which transient cycles are ignored
    analysis_start_time = (
        IGNORE_CYCLES * period
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    df = run_experiment(
        duration=duration,
        mass=MASS,
        activation_function=activation_sine,
        muscle_parameters={
            "f_max": F_MAX,
        }
    )

    all_runs[frequency] = df

    # --------------------------------------------------------
    # Save complete result
    # --------------------------------------------------------

    output_file = (
        RESULTS_PATH
        / f"frequency_{frequency:.2f}Hz.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Analyze steady-state cycles
    # --------------------------------------------------------

    steady = df[
        df["time"] >= analysis_start_time
    ]

    max_velocity = (
        steady["tendon_velocity"].max()
    )

    min_velocity = (
        steady["tendon_velocity"].min()
    )

    delta_velocity = (
        max_velocity
        - min_velocity
    )

    max_abs_velocity = (
        steady["tendon_velocity"]
        .abs()
        .max()
    )

    rms_velocity = np.sqrt(
        np.mean(
            steady["tendon_velocity"] ** 2
        )
    )

    summary.append(
        {
            "frequency": frequency,
            "period": period,

            "max_velocity":
                max_velocity,

            "min_velocity":
                min_velocity,

            "delta_velocity":
                delta_velocity,

            "max_abs_velocity":
                max_abs_velocity,

            "rms_velocity":
                rms_velocity,
        }
    )


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = (
    RESULTS_PATH
    / "frequency_velocity_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

# ============================================================
# Find critical frequency
# ============================================================

# First derivative:
# how much velocity changes per Hz
summary_df["velocity_slope"] = np.gradient(
    summary_df["delta_velocity"],
    summary_df["frequency"]
)

# Second derivative:
# how much the slope itself changes
summary_df["velocity_slope_change"] = np.gradient(
    summary_df["velocity_slope"],
    summary_df["frequency"]
)

# Avoid using boundary points because numerical
# derivatives are less reliable there
interior = summary_df.iloc[1:-1]

critical_freq_index = (
    interior["velocity_slope_change"]
    .abs()
    .idxmax()
)

critical_freq = summary_df.loc[
    critical_freq_index,
    "frequency"
]

print(
    f"\nApproximate critical freq per velocity analysis: "
    f"{critical_freq:.2f} Hz"
)


# ============================================================
# GRAPH 1
# Frequency vs velocity range
# ============================================================

fig1 = plt.figure()

plt.plot(
    summary_df["frequency"],
    summary_df["delta_velocity"],
    marker="o",
)

plt.xlabel(
    "Excitation Frequency [Hz]"
)

plt.ylabel(
    "Velocity Range Δv [m/s]"
)

plt.title(
    "Effect of Excitation Frequency on Tendon Velocity"
)

plt.grid(True)
plt.tight_layout()

fig1.savefig(
    RESULTS_PATH
    / "frequency_vs_delta_velocity.png",
    dpi=300,
)


# ============================================================
# GRAPH 2
# Velocity time series
# ============================================================

fig2 = plt.figure()

for frequency, df in all_runs.items():

    plt.plot(
        df["time"],
        df["tendon_velocity"],
        label=f"{frequency:.2f} Hz",
    )

plt.xlabel("Time [s]")
plt.ylabel("Tendon Velocity [m/s]")

plt.title(
    "Tendon Velocity at Different Excitation Frequencies"
)

plt.grid(True)

plt.legend(
    title="Frequency"
)

plt.tight_layout()

fig2.savefig(
    RESULTS_PATH
    / "frequency_velocity_time_series.png",
    dpi=300,
)

# ============================================================
# Generate report
# ============================================================

parameters = {
    "mass [kg]": MASS,
    "f_max [N]": F_MAX,
    "number_of_cycles": NUMBER_OF_CYCLES,
    "ignored_cycles": IGNORE_CYCLES,
    "activation_function": "sinusoidal",
    "minimum_activation": 0.01,
    "maximum_activation": 0.10,
}

key_results = {
    "Approximate critical frequency - velocity [Hz]":
        critical_freq,
}

figure_paths = [
    RESULTS_PATH
    / "frequency_vs_delta_velocity.png",

    RESULTS_PATH
    / "frequency_velocity_time_series.png",
]

document_experiment(
    title="PAM Excitation Frequency Test",

    description=(
        "Evaluation of tendon velocity response "
        "under sinusoidal activation signals with "
        "different excitation frequencies."
    ),

    parameters=parameters,

    summary_df=summary_df,

    key_results=key_results,

    figure_paths=figure_paths,

    results_directory=RESULTS_PATH,

    max_rows=20,

    filename_prefix="frequency_velocity_report",
)

# ============================================================
# Show all graphs simultaneously
# ============================================================
plt.show()
