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
    / "sensitivity_dse"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Experiment configuration
# ============================================================

# Nominal value from hill.json
DSE_NOMINAL = 0.9997764406809702

DSE_VALUES = {
    "-10%": 0.9 * DSE_NOMINAL,
    "nominal": DSE_NOMINAL,
    "+10%": 1.1 * DSE_NOMINAL,
}


# Keep the mechanical load fixed
MASS = 12.0

F_MAX = 1500.0


# Candidate frequencies
# Start with a reasonably narrow region.
# You can refine this later.
FREQUENCIES = np.arange(
    0.75,
    1.25,
    0.05
)


# Number of sinusoidal cycles simulated
NUMBER_OF_CYCLES = 5

# Ignore initial cycles when computing metrics
IGNORE_CYCLES = 2


# Activation signal limits
ACTIVATION_MEAN = 0.055
ACTIVATION_AMPLITUDE = 0.045


# ============================================================
# Storage
# ============================================================

summary = []

all_runs = {}


# ============================================================
# Experiments
# ============================================================

for frequency in FREQUENCIES:

    period = 1.0 / frequency

    duration = (
        NUMBER_OF_CYCLES
        * period
    )

    analysis_start_time = (
        IGNORE_CYCLES
        * period
    )

    print()
    print("=" * 70)
    print(
        f"Frequency = {frequency:.2f} Hz"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Activation function
    # --------------------------------------------------------

    def activation_sine(
        t,
        f=frequency,
    ):
        return (
            ACTIVATION_MEAN
            - ACTIVATION_AMPLITUDE
            * math.cos(
                2
                * math.pi
                * f
                * t
            )
        )

    # --------------------------------------------------------
    # Run -10%, nominal, +10%
    # --------------------------------------------------------

    for case, d_se in DSE_VALUES.items():

        print(
            f"  Running {case}: "
            f"d_se = {d_se:.6f}"
        )

        df = run_experiment(
            duration=duration,
            mass=MASS,
            activation_function=activation_sine,
            muscle_parameters={
                "f_max": F_MAX,
                "d_se": d_se,
            }
        )

        # Save complete run in memory
        all_runs[
            (frequency, case)
        ] = df

        # ----------------------------------------------------
        # Save raw CSV
        # ----------------------------------------------------

        csv_path = (
            RESULTS_PATH
            / (
                f"freq_{frequency:.2f}Hz_"
                f"dse_{case}.csv"
            )
        )

        df.to_csv(
            csv_path,
            index=False,
        )

        # ----------------------------------------------------
        # Analyze steady-state cycles
        # ----------------------------------------------------

        response = df[
            df["time"]
            >= analysis_start_time
        ]

        max_velocity = (
            response["tendon_velocity"]
            .max()
        )

        min_velocity = (
            response["tendon_velocity"]
            .min()
        )

        delta_velocity = (
            max_velocity
            - min_velocity
        )

        max_abs_velocity = (
            response["tendon_velocity"]
            .abs()
            .max()
        )

        rms_velocity = np.sqrt(
            np.mean(
                response[
                    "tendon_velocity"
                ] ** 2
            )
        )

        summary.append(
            {
                "frequency": frequency,
                "case": case,
                "d_se": d_se,
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
# Convert to DataFrame
# ============================================================

summary_df = pd.DataFrame(
    summary
)


# ============================================================
# Compute normalized sensitivity
# ============================================================

sensitivity_results = []


for frequency in FREQUENCIES:

    freq_data = summary_df[
        np.isclose(
            summary_df["frequency"],
            frequency,
        )
    ]

    minus = freq_data[
        freq_data["case"] == "-10%"
    ].iloc[0]

    nominal = freq_data[
        freq_data["case"] == "nominal"
    ].iloc[0]

    plus = freq_data[
        freq_data["case"] == "+10%"
    ].iloc[0]

    # --------------------------------------------------------
    # Normalized central sensitivity
    #
    # S =
    # ((y_plus - y_minus) / y_nominal)
    # /
    # ((p_plus - p_minus) / p_nominal)
    #
    # For ±10%:
    #
    # denominator = 0.20
    # --------------------------------------------------------

    if (
        abs(
            nominal[
                "delta_velocity"
            ]
        )
        > 1e-12
    ):

        sensitivity = (
            (
                plus["delta_velocity"]
                - minus[
                    "delta_velocity"
                ]
            )
            /
            nominal[
                "delta_velocity"
            ]
        ) / 0.20

    else:

        sensitivity = np.nan

    sensitivity_results.append(
        {
            "frequency": frequency,

            "delta_v_minus":
                minus[
                    "delta_velocity"
                ],

            "delta_v_nominal":
                nominal[
                    "delta_velocity"
                ],

            "delta_v_plus":
                plus[
                    "delta_velocity"
                ],

            "normalized_sensitivity":
                sensitivity,
        }
    )


sensitivity_df = pd.DataFrame(
    sensitivity_results
)


# ============================================================
# Identify frequency of maximum sensitivity
# ============================================================

valid_sensitivity = (
    sensitivity_df
    .dropna(
        subset=[
            "normalized_sensitivity"
        ]
    )
)


max_sensitivity_index = (
    valid_sensitivity[
        "normalized_sensitivity"
    ]
    .abs()
    .idxmax()
)


critical_frequency = (
    valid_sensitivity.loc[
        max_sensitivity_index,
        "frequency"
    ]
)


max_sensitivity = (
    valid_sensitivity.loc[
        max_sensitivity_index,
        "normalized_sensitivity"
    ]
)


print()
print("=" * 70)

print(
    "Maximum d_se sensitivity found at:"
)

print(
    f"  Frequency = "
    f"{critical_frequency:.2f} Hz"
)

print(
    f"  Normalized sensitivity = "
    f"{max_sensitivity:.6f}"
)

print("=" * 70)


# ============================================================
# Save summary files
# ============================================================

summary_df.to_csv(
    RESULTS_PATH
    / "dse_all_runs_summary.csv",
    index=False,
)

sensitivity_df.to_csv(
    RESULTS_PATH
    / "dse_sensitivity_summary.csv",
    index=False,
)


# ============================================================
# GRAPH 1
# Delta velocity vs frequency
# for the three d_se values
# ============================================================

fig1 = plt.figure()

for case in DSE_VALUES:

    case_data = summary_df[
        summary_df["case"] == case
    ]

    plt.plot(
        case_data["frequency"],
        case_data["delta_velocity"],
        marker="o",
        label=case,
    )


plt.xlabel(
    "Excitation Frequency [Hz]"
)

plt.ylabel(
    "Velocity Range Δv [m/s]"
)

plt.title(
    "Effect of d_se on Tendon Velocity Response"
)

plt.grid(True)

plt.legend(
    title="d_se variation"
)

plt.tight_layout()


FIG1_PATH = (
    RESULTS_PATH
    / "dse_frequency_delta_velocity.png"
)

fig1.savefig(
    FIG1_PATH,
    dpi=300,
)


# ============================================================
# GRAPH 2
# Normalized sensitivity vs frequency
# ============================================================

fig2 = plt.figure()

plt.plot(
    sensitivity_df[
        "frequency"
    ],
    sensitivity_df[
        "normalized_sensitivity"
    ],
    marker="o",
)


plt.axvline(
    x=critical_frequency,
    linestyle="--",
)


plt.xlabel(
    "Excitation Frequency [Hz]"
)

plt.ylabel(
    "Normalized Sensitivity"
)

plt.title(
    "Sensitivity of Tendon Velocity to d_se"
)

plt.grid(True)

plt.tight_layout()


FIG2_PATH = (
    RESULTS_PATH
    / "dse_sensitivity_vs_frequency.png"
)

fig2.savefig(
    FIG2_PATH,
    dpi=300,
)


# ============================================================
# GRAPH 3
# Velocity time series at the most sensitive frequency
# ============================================================

fig3 = plt.figure()


for case in DSE_VALUES:

    df = all_runs[
        (
            critical_frequency,
            case,
        )
    ]

    plt.plot(
        df["time"],
        df["tendon_velocity"],
        label=case,
    )


plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Tendon Velocity [m/s]"
)

plt.title(
    (
        "Tendon Velocity at "
        f"{critical_frequency:.2f} Hz"
    )
)

plt.grid(True)

plt.legend(
    title="d_se variation"
)

plt.tight_layout()


FIG3_PATH = (
    RESULTS_PATH
    / "dse_velocity_time_series.png"
)

fig3.savefig(
    FIG3_PATH,
    dpi=300,
)


# ============================================================
# Generate automatic report
# ============================================================

parameters = {
    "nominal d_se":
        DSE_NOMINAL,

    "d_se -10%":
        DSE_VALUES["-10%"],

    "d_se +10%":
        DSE_VALUES["+10%"],

    "mass [kg]":
        MASS,

    "f_max [N]":
        F_MAX,

    "frequency min [Hz]":
        FREQUENCIES.min(),

    "frequency max [Hz]":
        FREQUENCIES.max(),

    "frequency step [Hz]":
        FREQUENCIES[1]
        - FREQUENCIES[0],

    "number of cycles":
        NUMBER_OF_CYCLES,

    "ignored cycles":
        IGNORE_CYCLES,

    "activation mean":
        ACTIVATION_MEAN,

    "activation amplitude":
        ACTIVATION_AMPLITUDE,
}


key_results = {
    "Frequency of maximum sensitivity [Hz]":
        critical_frequency,

    "Maximum normalized sensitivity":
        max_sensitivity,
}


document_experiment(
    title=(
        "PAM Serial Damping "
        "Sensitivity Analysis"
    ),

    description=(
        "Local sensitivity analysis of the "
        "serial damping parameter d_se. "
        "The parameter was perturbed by ±10% "
        "around its nominal value while the "
        "excitation frequency was varied. "
        "The output metric was the tendon "
        "velocity range Δv."
    ),

    parameters=parameters,

    summary_df=sensitivity_df,

    key_results=key_results,

    figure_paths=[
        FIG1_PATH,
        FIG2_PATH,
        FIG3_PATH,
    ],

    results_directory=
        RESULTS_PATH,

    max_rows=20,

    filename_prefix=
        "dse_sensitivity_report",
)


# ============================================================
# Show all figures simultaneously
# ============================================================

plt.show()
