from pathlib import Path
from pam_model import create_default_hill_pam
from pam_model import create_hill_pam_from_json
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "single_pam.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

muscle = create_hill_pam_from_json(
    ROOT / "config" / "hill.json",
    a_init=0.01,
)

slider_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "slider"
)

dof_id = model.jnt_dofadr[slider_id]

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

mujoco.mj_forward(model, data)
initial_tendon_length = data.sensordata[length_adr]

test_force, test_dl_ce = muscle.compute(
    l_mtc=0.0,
    dot_l_mtc=0.0,
    activation=0.01,
)

print(
    f"Initial Hill force: {test_force:.3f} N"
)
print(
    f"Initial dL_CE: {test_dl_ce:.9f} m/s"
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        tendon_length = data.sensordata[length_adr]
        tendon_velocity = data.sensordata[velocity_adr]

        # Change relative to the initial MuJoCo configuration
        delta_l_mtc = tendon_length - initial_tendon_length

        if data.time < 3.0:
            activation = 0.01
        else:
            activation = 0.10

        force = muscle.step(
            activation=activation,
            l_mtc=delta_l_mtc,
            dot_l_mtc=tendon_velocity,
            dt=model.opt.timestep,
        )

        data.qfrc_applied[dof_id] = force

        mujoco.mj_step(model, data)

        viewer.sync()

        q = data.qpos[model.jnt_qposadr[slider_id]]

        print(
            f"t={data.time:.3f} | "
            f"q={q:.5f} | "
            f"L={tendon_length:.4f} | "
            f"dL={delta_l_mtc:.4f} | "
            f"v={tendon_velocity:.4f} | "
            f"a={activation:.3f} | "
            f"F={force:.2f}"
        )

        time.sleep(model.opt.timestep)
