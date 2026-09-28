from pathlib import Path
import time
import mujoco.viewer
import mujoco
import numpy as np
import pandas as pd
import math

from pam_model import create_hill_pam_from_json


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "single_pam.xml"
CONFIG_PATH = ROOT / "config" / "hill.json"
RESULTS_PATH = ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)


def run_experiment(
    duration=3.0,
    mass=1.0,
    activation_function=None,
    muscle_parameters=None,
    show_viewer=True,
):
    """
    Run one PAM simulation experiment.

    Parameters
    ----------
    duration : float
        Simulation duration [s].

    mass : float
        Moving body mass [kg].

    activation_function : callable
        Function activation_function(t) -> activation.

    muscle_parameters : dict
        Optional Hill-model parameter overrides.

        Supported parameters:
            f_max
            l_CEopt
            a_rel0
            b_rel0
            d_se
            r_se
            delta_f

    Returns
    -------
    pandas.DataFrame
        Time-series simulation data.
    """

    if activation_function is None:
        activation_function = lambda t: 0.01

    if muscle_parameters is None:
        muscle_parameters = {}

    ###########################################################
    # Load MuJoCo
    ###########################################################

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    ###########################################################
    # Change moving mass
    ###########################################################

    load_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "load"
    )

    model.body_mass[load_body_id] = mass

    ###########################################################
    # Joint IDs
    ###########################################################

    slider_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "slider"
    )

    dof_id = model.jnt_dofadr[slider_id]
    qpos_id = model.jnt_qposadr[slider_id]

    ###########################################################
    # Sensor addresses
    ###########################################################

    length_sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "pam_length"
    )

    velocity_sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "pam_velocity"
    )

    length_adr = model.sensor_adr[length_sensor_id]
    velocity_adr = model.sensor_adr[velocity_sensor_id]

    ###########################################################
    # Build muscle
    ###########################################################

    initial_activation = activation_function(0.0)

    muscle = create_hill_pam_from_json(
        CONFIG_PATH,
        a_init=initial_activation,
        overrides=muscle_parameters,
    )

    ###########################################################
    # Initialize simulation
    ###########################################################

    mujoco.mj_forward(model, data)

    initial_tendon_length = data.sensordata[length_adr]

    records = []

    def simulation_step():

        tendon_length = data.sensordata[length_adr]
        tendon_velocity = data.sensordata[velocity_adr]

        delta_l_mtc = (
            tendon_length
            - initial_tendon_length
        )

        activation = activation_function(data.time)

        force = muscle.step(
            activation=activation,
            l_mtc=delta_l_mtc,
            dot_l_mtc=tendon_velocity,
            dt=model.opt.timestep,
        )

        data.qfrc_applied[dof_id] = force

        records.append(
            {
                "time": data.time,
                "position": data.qpos[qpos_id],
                "joint_velocity": data.qvel[dof_id],
                "tendon_length": tendon_length,
                "delta_l_mtc": delta_l_mtc,
                "tendon_contraction": -delta_l_mtc,
                "tendon_velocity": tendon_velocity,
                "activation": activation,
                "force": force,
                "l_ce": muscle.l_ce_state,
                "mass": mass,
            }
        )

        mujoco.mj_step(model, data)

    ###########################################################
    # Simulation
    ###########################################################

    if show_viewer:

        with mujoco.viewer.launch_passive(model, data) as viewer:

            while viewer.is_running() and data.time < duration:

                step_start = time.time()

                simulation_step()

                viewer.sync()

                # Run approximately in real time
                elapsed = time.time() - step_start
                sleep_time = model.opt.timestep - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

    else:

        while data.time < duration:
            simulation_step()

    return pd.DataFrame(records)


if __name__ == "__main__":

    def activation_step(t):
        if t < 2.0:
            return 0.01
        else:
            return 0.10

    def activation_slow(t):
        if t < 2.0:
            return 0.10
        else:
            return 0.055 - 0.045 * math.cos(2*math.pi*0.25*t)

    df = run_experiment(
        duration=6.0,
        mass=1.0,
        activation_function=activation_step,
        muscle_parameters={
            "f_max": 1650.0
        }
    )

    output = RESULTS_PATH / "activation_step.csv"

    df.to_csv(output, index=False)

    print(df)
    print(f"\nSaved to: {output}")
