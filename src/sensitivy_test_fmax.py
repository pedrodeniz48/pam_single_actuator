from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from experiments import run_experiment
import activation_funcs as act
from report_utils import document_experiment


ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "sensitivity_fmax"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


FMAX_NOMINAL = 1500.0

FMAX_VALUES = {
    "-10%": 0.9 * FMAX_NOMINAL,
    "nominal": FMAX_NOMINAL,
    "+10%": 1.1 * FMAX_NOMINAL,
}

OPERATING_MASS = 12.0  # 14.0
DURATION = 5.0
ANALYSIS_START_TIME = 0.0


summary = []
all_runs = {}


for label, fmax in FMAX_VALUES.items():

    print(
        f"Running {label}: "
        f"f_max = {fmax:.1f} N"
    )

    df = run_experiment(
        duration=DURATION,
        mass=OPERATING_MASS,
        activation_function=act.step,
        muscle_parameters={
            "f_max": fmax,
        }
    )

    all_runs[label] = df

    response = df[
        df["time"] >= ANALYSIS_START_TIME
    ]

    max_contraction = (
        response["tendon_contraction"]
        .max()
    )

    max_velocity = (
        response["tendon_velocity"]
        .abs()
        .max()
    )

    summary.append(
        {
            "case": label,
            "f_max": fmax,
            "max_contraction": max_contraction,
            "max_tendon_velocity": max_velocity,
        }
    )


summary_df = pd.DataFrame(summary)

minus = summary_df[
    summary_df["case"] == "-10%"
].iloc[0]

nominal = summary_df[
    summary_df["case"] == "nominal"
].iloc[0]

plus = summary_df[
    summary_df["case"] == "+10%"
].iloc[0]

sensitivity_contraction = (
    (
        plus["max_contraction"]
        - minus["max_contraction"]
    )
    / nominal["max_contraction"]
) / 0.2

sensitivity_velocity = (
    (
        plus["max_tendon_velocity"]
        - minus["max_tendon_velocity"]
    )
    / nominal["max_tendon_velocity"]
) / 0.2

fig1 = plt.figure()

for label, df in all_runs.items():
    plt.plot(
        df["time"],
        df["tendon_contraction"],
        label=label,
    )

plt.xlabel("Time [s]")
plt.ylabel("Tendon Contraction [m]")
plt.title(
    f"Sensitivity to f_max at "
    f"{OPERATING_MASS:.2f} kg"
)
plt.legend()
plt.grid(True)
plt.tight_layout()

FIG1_PATH = (
    RESULTS_PATH
    / "sentivity_fmax.png"
)

fig1.savefig(
    FIG1_PATH,
    dpi=300,
)

fig2 = plt.figure()

plt.plot(
    summary_df["f_max"],
    summary_df["max_contraction"],
    marker="o",
)

plt.xlabel("f_max [N]")
plt.ylabel("Maximum Tendon Contraction [m]")
plt.title("Local Sensitivity to f_max")
plt.grid(True)
plt.tight_layout()

FIG2_PATH = (
    RESULTS_PATH
    / "max_tendon_contraction.png"
)

fig2.savefig(
    FIG2_PATH,
    dpi=300,
)

# ============================================================
# Save numerical summary
# ============================================================

SUMMARY_PATH = (
    RESULTS_PATH
    / "fmax_sensitivity_summary.csv"
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Report configuration
# ============================================================

parameters = {
    "Operating mass [kg]":
        OPERATING_MASS,

    "Nominal f_max [N]":
        FMAX_NOMINAL,

    "f_max -10% [N]":
        FMAX_VALUES["-10%"],

    "f_max nominal [N]":
        FMAX_VALUES["nominal"],

    "f_max +10% [N]":
        FMAX_VALUES["+10%"],

    "Duration [s]":
        DURATION,

    "Analysis start time [s]":
        ANALYSIS_START_TIME,

    "Activation function":
        "step",
}


key_results = {
    "Normalized sensitivity - contraction":
        sensitivity_contraction,

    "Normalized sensitivity - velocity":
        sensitivity_velocity,

    "Nominal maximum contraction [m]":
        nominal["max_contraction"],

    "Nominal maximum tendon velocity [m/s]":
        nominal["max_tendon_velocity"],
}


figure_paths = [
    FIG1_PATH,
    FIG2_PATH,
]


# ============================================================
# Generate PDF report
# ============================================================

document_experiment(
    title="PAM Maximum Force Sensitivity Analysis",

    description=(
        "Local sensitivity analysis of the contractile "
        "maximum force parameter f_max. "
        "The parameter was varied by ±10% around its "
        "nominal value while maintaining a fixed load mass. "
        "Maximum tendon contraction and maximum absolute "
        "tendon velocity were used as output metrics."
    ),

    parameters=parameters,

    summary_df=summary_df,

    key_results=key_results,

    figure_paths=figure_paths,

    results_directory=RESULTS_PATH,

    max_rows=10,

    filename_prefix="fmax_sensitivity_report",
)

plt.show()
