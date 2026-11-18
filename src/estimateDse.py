"""Estimate d_se using a 1 Hz sinusoidal excitation selected in the dynamic study.

Run from the project folder: python src/estimateDse.py

It estimates one unknown parameter (serial damping d_se); all other model parameters remain known.
The reference data and fitted model use the same simulator, so this is a
synthetic parameter-recovery experiment, not validation on a real actuator.
"""

from pathlib import Path
from datetime import datetime
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Import experiments and the activation functions
import experiments
import activation_funcs as act


#1.EXPERIMENT SETTINGS
# Use a load mass in the non-saturated velocity regime (11.0 kg) and a 1 Hz sinusoidal
# activation signal so velocity-dependent serial damping (d_se) is clearly observable.
MASS = 12.0                  # Load mass [kg] (high velocity regime).
TRUE_DSE = 0.9998            # Known ground truth for synthetic data (nominal config).
DURATION = 5.0               # Simulation duration [s].
FIT_START = 0.0              # Fit from t=0 s to capture all dynamic velocity transients.

#Sinusoidal excitation settings (frequency = 1.0 Hz, activation range [0.01, 0.10])
FREQ_HZ = 1.0             # Excitation frequency [Hz].
ACT_MIN = 0.01               # Minimum activation level [-].
ACT_MAX = 0.10               # Maximum activation level [-].

#Search limits and initial guesses for d_se
LOWER_DSE = 0.1
UPPER_DSE = 3.0
GRID_POINTS = 41
INITIAL_GUESSES = [0.5, 1.5]

#Noise info
NOISE_STD = 0.0001           # Standard deviation: 0.0001 m = 0.1 mm.
#NOISE_STD = 0.001          # Stress-test noise: 0.001 m = 1.0 mm.
NUMBER_OF_RUNS = 20          # Independent noise sequences, same noise level.
FIRST_SEED = 0               # Fixed seeds make the results reproducible.

SHOW_PLOTS = True

ROOT = Path(__file__).resolve().parent.parent


#2.ACTIVATION & HELPER FUNCTIONS

def sin_1hz_activation(t):
    """Sinusoidal activation at 1 Hz oscillating smoothly between ACT_MIN and ACT_MAX."""
    a0 = 0.5 * (ACT_MAX + ACT_MIN)
    amp = 0.5 * (ACT_MAX - ACT_MIN)
    return a0 - amp * np.cos(2.0 * np.pi * FREQ_HZ * t)


def simulate(dse, activation):
    """Run a fresh simulation with only d_se changed."""
    data = experiments.run_experiment(
        duration=DURATION,
        mass=MASS,
        activation_function=activation,
        muscle_parameters={"d_se": float(dse)},
    )
    if data.empty or not np.isfinite(data[["time", "tendon_length"]]).all().all():
        raise ValueError("The simulation returned invalid data.")
    return data


def get_length(data, expected_time):
    """Check that prediction and measurements use the same observation times."""
    time = data["time"].to_numpy()
    if time.shape != expected_time.shape or not np.allclose(
        time, expected_time, rtol=0, atol=1e-12
    ):
        raise ValueError("Prediction and measurement time grids do not match.")
    return data["tendon_length"].to_numpy()


def rmse_mm(error):
    """Convert an error vector in metres to RMSE in millimetres."""
    return float(1000 * np.sqrt(np.mean(error ** 2)))


#3.ESTIMATE THE PARAMETER FROM ONE MEASUREMENT SEQUENCE

def estimate_dse(measured, fit_mask, predict):
    """Fit directly with nonlinear least squares, starting from two guesses."""
    best_dse = None
    best_cost = np.inf
    best_source = None

    def residuals(parameters):
        #Scaling to mm improves numerical conditioning without changing the minimum.
        predicted = predict(float(parameters[0]))
        return 1000 * (predicted[fit_mask] - measured[fit_mask])

    attempts = []
    for initial_guess in INITIAL_GUESSES:
        source = f"initial_{initial_guess:g}"
        result = least_squares(
            residuals,
            x0=[initial_guess],
            bounds=([LOWER_DSE], [UPPER_DSE]),
            x_scale=[1.0],
            diff_step=1e-4,
            max_nfev=100,
        )
        cost = float(np.mean(result.fun ** 2))
        attempts.append({
            "source": source,
            "initial_dse": initial_guess,
            "estimated_dse": float(result.x[0]),
            "MSE_mm2": cost,
            "success": bool(result.success),
            "active_bound": int(result.active_mask[0]),
            "message": str(result.message),
        })
        #Select using identification error only.
        if result.success and np.isfinite(cost) and cost < best_cost:
            best_dse = float(result.x[0])
            best_cost = cost
            best_source = source

    if best_dse is None:
        messages = "; ".join(attempt["message"] for attempt in attempts)
        raise RuntimeError(f"No successful least-squares solution. {messages}")

    return best_dse, best_source, attempts


def main():

    #4.PREPARE OUTPUT FOLDER AND CLEAN REFERENCE DATA

    if NUMBER_OF_RUNS < 2 or NOISE_STD <= 0:
        raise ValueError("Use at least two noise runs and positive noise STD.")
    if not 0 <= FIT_START < DURATION:
        raise ValueError("Check FIT_START and DURATION.")
    if GRID_POINTS < 3 or not 0 < LOWER_DSE < UPPER_DSE:
        raise ValueError("Check the parameter grid and bounds.")
    if any(not LOWER_DSE < guess < UPPER_DSE for guess in INITIAL_GUESSES):
        raise ValueError("Initial guesses must lie inside the bounds.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ROOT / "results" / "dse_estimation" / timestamp
    folder.mkdir(parents=True, exist_ok=False)

    #Save settings and source files so the experiment can be reproduced.
    settings = {name: value for name, value in globals().items()
                if name.isupper() and isinstance(value, (int, float, bool, list))}
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    source_folder = folder / "source_files"
    source_folder.mkdir()
    for path in [Path(__file__), Path(experiments.__file__), Path(act.__file__),
                 Path(experiments.__file__).parent / "pam_model.py",
                 Path(experiments.CONFIG_PATH), Path(experiments.MODEL_PATH)]:
        if path.is_file():
            shutil.copy2(path, source_folder / path.name)

    #Identification uses the 1.0 Hz sinusoidal activation.
    reference = simulate(TRUE_DSE, sin_1hz_activation)
    time = reference["time"].to_numpy()
    clean_length = reference["tendon_length"].to_numpy()
    fit_mask = time >= FIT_START
    reference.to_csv(folder / "identification_reference.csv", index=False)

    #Validation uses an unseen excitation trajectory (step input).
    validation = simulate(TRUE_DSE, act.step)
    validation_time = validation["time"].to_numpy()
    validation_length = validation["tendon_length"].to_numpy()
    validation.to_csv(folder / "validation_reference.csv", index=False)

    #Reuse deterministic candidate simulations to reduce computation time.
    predictions = {}
    def predict(dse):
        if dse not in predictions:
            predictions[dse] = get_length(simulate(dse, sin_1hz_activation), time)
        return predictions[dse]

    grid = np.linspace(LOWER_DSE, UPPER_DSE, GRID_POINTS)
    grid_predictions = []
    for index, candidate in enumerate(grid):
        print(f"Preparing grid: {index + 1}/{GRID_POINTS}", flush=True)
        grid_predictions.append(predict(float(candidate)))
    grid_predictions = np.array(grid_predictions)

    #5.RUN THE CLEAN CASE AND REPEATED NOISY CASES

    results = []
    all_attempts = []
    examples = []
    seeds = [None] + list(range(FIRST_SEED, FIRST_SEED + NUMBER_OF_RUNS))

    for seed in seeds:
        if seed is None:
            label = "without_noise"
            measured = clean_length.copy()
        else:
            label = f"noise_seed_{seed}"
            generator = np.random.default_rng(seed)
            noise = generator.normal(0, NOISE_STD, len(time))
            measured = clean_length + noise

        print(f"Estimating d_se (f = {FREQ_HZ} Hz): {label}", flush=True)
        estimated, source, attempts = estimate_dse(
            measured, fit_mask, predict
        )
        predicted = predict(estimated)

        #Validate the final estimate on the unseen step input, without refitting.
        validation_prediction = get_length(
            simulate(estimated, act.step), validation_time
        )
        signed_error = 100 * (estimated - TRUE_DSE) / TRUE_DSE
        at_bound = (np.isclose(estimated, LOWER_DSE, rtol=0, atol=1e-5) or
                    np.isclose(estimated, UPPER_DSE, rtol=0, atol=1e-5))
        results.append({
            "case": label, "seed": seed,
            "estimated_dse": estimated,
            "signed_error_percent": signed_error,
            "absolute_error_percent": abs(signed_error),
            "fit_RMSE_mm": rmse_mm((predicted - measured)[fit_mask]),
            "validation_RMSE_mm": rmse_mm(validation_prediction - validation_length),
            "selected_source": source, "selected_at_bound": bool(at_bound),
        })
        for attempt in attempts:
            all_attempts.append({"case": label, **attempt})
        print(f"  Estimated d_se: {estimated:.6f}; error: {abs(signed_error):.6f}%")

        # Save every realization
        pd.DataFrame({"time_s": time, "clean_m": clean_length,
                      "measured_m": measured, "predicted_m": predicted,
                      "used_for_fit": fit_mask}).to_csv(folder / f"fit_{label}.csv", index=False)
        pd.DataFrame({"time_s": validation_time, "reference_m": validation_length,
                      "predicted_m": validation_prediction}).to_csv(folder / f"validation_{label}.csv", index=False)
        pd.DataFrame(results).to_csv(folder / "all_estimates.csv", index=False)
        pd.DataFrame(all_attempts).to_csv(folder / "optimization_attempts.csv", index=False)

        if seed is None or seed == FIRST_SEED:
            examples.append((label, measured, predicted, validation_prediction))

    #6.SUMMARIZE PARAMETER ACCURACY AND VARIABILITY

    table = pd.DataFrame(results)
    noisy = table[table["case"] != "without_noise"]
    summary = pd.DataFrame([{
        "freq_Hz": FREQ_HZ,
        "noise_std_mm": NOISE_STD * 1000,
        "number_of_runs": NUMBER_OF_RUNS,
        "mean_dse": noisy["estimated_dse"].mean(),
        "std_dse": noisy["estimated_dse"].std(ddof=1),
        "mean_signed_error_percent": noisy["signed_error_percent"].mean(),
        "mean_absolute_error_percent": noisy["absolute_error_percent"].mean(),
        "max_absolute_error_percent": noisy["absolute_error_percent"].max(),
        "mean_fit_RMSE_mm": noisy["fit_RMSE_mm"].mean(),
        "mean_validation_RMSE_mm": noisy["validation_RMSE_mm"].mean(),
        "selected_bound_count": int(noisy["selected_at_bound"].sum()),
    }])
    summary.to_csv(folder / "noise_summary.csv", index=False)
    print("\nResults from all runs:")
    print(table.to_string(index=False))
    print("\nSummary across noise realizations:")
    print(summary.to_string(index=False))

    #7.PLOT FIT, RESIDUALS, VALIDATION AND REPEATED ESTIMATES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (label, measured, predicted, val_predicted) in enumerate(examples):
        axes[row, 0].plot(time, measured, color="grey", label="Measurements")
        axes[row, 0].plot(time, predicted, label="Estimated model")
        if FIT_START > 0:
            axes[row, 0].axvline(FIT_START, color="black", linestyle=":")
        axes[row, 0].set(title=f"Fit (f={FREQ_HZ} Hz): {label}", xlabel="Time [s]", ylabel="Length [m]")
        axes[row, 1].plot(time[fit_mask], 1000 * (predicted - measured)[fit_mask])
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 1].set(title=f"Residuals: {label}", xlabel="Time [s]", ylabel="Residual [mm]")
        axes[row, 2].plot(validation_time, validation_length, color="black", label="Clean reference")
        axes[row, 2].plot(validation_time, val_predicted, "--", label="Estimated model")
        axes[row, 2].set(title="Validation: step input", xlabel="Time [s]", ylabel="Length [m]")
        axes[row, 0].legend()
        axes[row, 2].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "fit_residuals_validation.png", dpi=200)

    fig_noise, axes_noise = plt.subplots(1, 2, figsize=(11, 4))
    axes_noise[0].plot(noisy["seed"], noisy["estimated_dse"], "o-")
    axes_noise[0].axhline(TRUE_DSE, color="black", linestyle=":", label="True value")
    axes_noise[0].set(xlabel="Noise seed", ylabel="Estimated d_se [-]")
    axes_noise[0].legend()
    axes_noise[1].plot(noisy["seed"], noisy["absolute_error_percent"], "o-")
    axes_noise[1].set(xlabel="Noise seed", ylabel="Absolute parameter error [%]")
    for axis in axes_noise:
        axis.grid(True, alpha=0.3)
    fig_noise.tight_layout()
    fig_noise.savefig(folder / "noise_repetitions.png", dpi=200)

    print(f"\nResults saved to: {folder}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    plt.close(fig_noise)


if __name__ == "__main__":
    main()