from pathlib import Path
from pam_model import create_hill_pam_from_json
from pam_model import HillPAM
from pam_model import (
    ContractileElement,
    ParallelElasticElement,
    SerialElasticElement,
    SerialDampingElement,
)


ce = ContractileElement(
    f_max=1420,
    l_ce_opt=0.092,
    delta_w_limb_desc=0.35,
    delta_w_limb_asc=0.35,
    limb_desc=1.5,
    limb_asc=3.0,
    a_rel0=0.25,
    b_rel0=2.25,
    s_eccentric=2.0,
    c_eccentric=1.5,
)

pee = ParallelElasticElement(
    contractile_element=ce,
    L=0.9,
    v=2.5,
    F=2.0,
)

see = SerialElasticElement(
    l_0=0.172,
    delta_u_nll=0.0425,
    delta_u_l=0.017,
    delta_f=568,
)

sde = SerialDampingElement(
    contractile_element=ce,
    d_se=0.3,
    r_se=0.01,
)

print("SEE")
for l_mtc in [0.24, 0.264, 0.28, 0.30]:
    l_ce = 0.092

    f_see = see.get_force(
        l_mtc=l_mtc,
        l_ce=l_ce,
    )

    print(
        f"l_mtc={l_mtc:.3f} | "
        f"l_see={abs(l_mtc-l_ce):.3f} | "
        f"F_SEE={f_see:.3f} N"
    )

print("\nSDE")

f_ce = 500.0
f_pee = 50.0

for dot_l_mtc, dot_l_ce in [
    (0.0, 0.0),
    (0.1, 0.0),
    (0.0, 0.1),
    (0.1, 0.05),
]:
    f_sde = sde.get_force(
        f_ce=f_ce,
        f_pee=f_pee,
        f_max=ce.f_max,
        dot_l_mtc=dot_l_mtc,
        dot_l_ce=dot_l_ce,
    )

    print(
        f"dL_MTC={dot_l_mtc:.3f} | "
        f"dL_CE={dot_l_ce:.3f} | "
        f"F_SDE={f_sde:.3f} N"
    )

muscle = HillPAM(
    contractile_element=ce,
    parallel_elastic=pee,
    serial_damping=sde,
    serial_elastic=see,
    a_init=0.01,
    l_mtc_change_init=0.0,
    length=0.264,
)

print("\nFULL HILL MODEL")
print(
    f"Initial l_CE = "
    f"{muscle.l_ce_init:.6f} m"
)

for activation in [
    0.01,
    0.1,
    0.3,
    0.5,
]:
    force, dot_l_ce = muscle.compute(
        l_mtc=0.0,
        dot_l_mtc=0.0,
        activation=activation,
    )

    print(
        f"a={activation:.2f} | "
        f"F={force:.3f} N | "
        f"dL_CE={dot_l_ce:.6f} m/s"
    )

ROOT = Path(__file__).resolve().parent.parent

muscle = create_hill_pam_from_json(
    ROOT / "config" / "hill_test.json",
    a_init=0.01,
)
