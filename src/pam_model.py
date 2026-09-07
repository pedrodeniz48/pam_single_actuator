import math, json


class ContractileElement:
    def __init__(
        self,
        f_max,
        l_ce_opt,
        delta_w_limb_desc,
        delta_w_limb_asc,
        limb_desc,
        limb_asc,
        a_rel0,
        b_rel0,
        s_eccentric,
        c_eccentric,
    ):
        self.f_max = f_max
        self.l_ce_opt = l_ce_opt
        self.delta_w_limb_desc = delta_w_limb_desc
        self.delta_w_limb_asc = delta_w_limb_asc
        self.limb_desc = limb_desc
        self.limb_asc = limb_asc
        self.a_rel0 = a_rel0
        self.b_rel0 = b_rel0
        self.s_eccentric = s_eccentric
        self.c_eccentric = c_eccentric

    def get_isometric_force(self, l_ce):
        if l_ce >= self.l_ce_opt:
            return math.exp(
                -abs(
                    ((l_ce / self.l_ce_opt) - 1)
                    / self.delta_w_limb_desc
                ) ** self.limb_desc
            )

        return math.exp(
            -abs(
                ((l_ce / self.l_ce_opt) - 1)
                / self.delta_w_limb_asc
            ) ** self.limb_asc
        )

    def get_a_relative(self, l_ce, f_isom, activation):
        length_factor = 1.0 if l_ce < self.l_ce_opt else f_isom

        return (
            length_factor
            * self.a_rel0
            * 0.25
            * (1 + 3 * activation)
        )

    def get_b_relative(self, activation):
        return (
            self.b_rel0
            * (1.0 / 7.0)
            * (3 + 4 * activation)
        )


class ParallelElasticElement:
    def __init__(self, contractile_element, L, v, F):
        self.L = L
        self.v = v
        self.F = F

        self.l_pee0 = L * contractile_element.l_ce_opt

        self.k_pee = (
            F
            * (
                contractile_element.f_max
                /
                (
                    contractile_element.l_ce_opt
                    * (
                        contractile_element.delta_w_limb_desc
                        + 1
                        - L
                    )
                ) ** v
            )
        )

    def get_force(self, l_ce):
        if l_ce >= self.l_pee0:
            return self.k_pee * (l_ce - self.l_pee0) ** self.v

        return 0.0

class SerialElasticElement:
    def __init__(
        self,
        l_0,
        delta_u_nll,
        delta_u_l,
        delta_f,
    ):
        self.l_0 = l_0
        self.delta_u_nll = delta_u_nll
        self.delta_u_l = delta_u_l
        self.delta_f = delta_f

        self.l_nll = (1.0 + delta_u_nll) * l_0
        self.v = delta_u_nll / delta_u_l

        self.k_nl = (
            delta_f
            / (delta_u_nll * l_0) ** self.v
        )

        self.k_l = (
            delta_f
            / (delta_u_l * l_0)
        )

    def get_force(self, l_mtc, l_ce):
        l_see = abs(l_mtc - l_ce)

        if l_see >= self.l_nll:
            return (
                self.delta_f
                + self.k_l * (l_see - self.l_nll)
            )

        if self.l_0 < l_see < self.l_nll:
            return (
                self.k_nl
                * (l_see - self.l_0) ** self.v
            )

        return 0.0


class SerialDampingElement:
    def __init__(
        self,
        contractile_element,
        d_se,
        r_se,
    ):
        self.d_se = d_se
        self.r_se = r_se

        self.d_se_max = (
            d_se
            * (
                contractile_element.f_max
                * contractile_element.a_rel0
            )
            / (
                contractile_element.l_ce_opt
                * contractile_element.b_rel0
            )
        )

    def get_force(
        self,
        f_ce,
        f_pee,
        f_max,
        dot_l_mtc,
        dot_l_ce,
    ):
        t1 = (
            (1.0 - self.r_se)
            * ((f_ce + f_pee) / f_max)
        )

        t2 = dot_l_mtc - dot_l_ce

        return (
            self.d_se_max
            * (t1 + self.r_se)
            * t2
        )

class HillPAM:
    def __init__(
        self,
        contractile_element,
        parallel_elastic,
        serial_damping,
        serial_elastic,
        a_init,
        l_mtc_change_init,
        length,
    ):
        self.ce = contractile_element
        self.pee = parallel_elastic
        self.sde = serial_damping
        self.see = serial_elastic

        self.a_init = a_init

        self.l_mtc_init = length + l_mtc_change_init

        self.l_ce_init = self._find_initial_l_ce()

        # Internal CE displacement relative to initial equilibrium
        self.l_ce_state = 0.0

    def _initial_force_equilibrium(self, l_ce):
        l_mtc = self.l_mtc_init

        f_isom = self.ce.get_isometric_force(l_ce)
        f_pee = self.pee.get_force(l_ce)
        f_see = self.see.get_force(l_mtc, l_ce)

        f_ce = (
            self.ce.f_max
            * self.a_init
            * f_isom
        )

        return f_see - f_ce - f_pee

    def _find_initial_l_ce(self):
        """
        Simple bisection replacement for the Brent solver used
        by the original C++ implementation.
        """

        lower = 0.0
        upper = self.l_mtc_init

        f_lower = self._initial_force_equilibrium(lower)
        f_upper = self._initial_force_equilibrium(upper)

        if f_lower * f_upper > 0:
            raise ValueError(
                "Initial CE equilibrium is not bracketed."
            )

        for _ in range(100):
            middle = 0.5 * (lower + upper)
            f_middle = self._initial_force_equilibrium(middle)

            if abs(f_middle) < 1e-10:
                return middle

            if f_lower * f_middle <= 0:
                upper = middle
                f_upper = f_middle
            else:
                lower = middle
                f_lower = f_middle

        return 0.5 * (lower + upper)

    def compute(self, l_mtc, dot_l_mtc, activation):
        # Original code adds initial offsets
        l_mtc_abs = l_mtc + self.l_mtc_init
        l_ce_abs = self.l_ce_state + self.l_ce_init

        f_isom = self.ce.get_isometric_force(l_ce_abs)
        f_pee = self.pee.get_force(l_ce_abs)
        f_see = self.see.get_force(
            l_mtc_abs,
            l_ce_abs,
        )

        a_rel = self.ce.get_a_relative(
            l_ce_abs,
            f_isom,
            activation,
        )

        b_rel = self.ce.get_b_relative(
            activation,
        )

        d0 = (
            self.ce.l_ce_opt
            * b_rel
            * self.sde.d_se_max
            * (
                self.sde.r_se
                + (1.0 - self.sde.r_se)
                * (
                    activation * f_isom
                    + f_pee / self.ce.f_max
                )
            )
        )

        c2 = (
            self.sde.d_se_max
            * (
                self.sde.r_se
                - (
                    a_rel
                    - f_pee / self.ce.f_max
                )
                * (1.0 - self.sde.r_se)
            )
        )

        c1 = -(
            c2 * dot_l_mtc
            + d0
            + f_see
            - f_pee
            + self.ce.f_max * a_rel
        )

        fs = (
            f_see
            - f_pee
            - self.ce.f_max
            * activation
            * f_isom
        )

        c0 = (
            d0 * dot_l_mtc
            + self.ce.l_ce_opt
            * b_rel
            * fs
        )

        discriminant = (
            c1 * c1
            - 4.0 * c2 * c0
        )

        if discriminant < 0:
            discriminant = 0.0

        dot_l_ce = (
            -c1
            - math.sqrt(discriminant)
        ) / (2.0 * c2)

        # Eccentric contraction
        if dot_l_ce > 0:
            a_rel_con = a_rel
            b_rel_con = b_rel

            a_rel = (
                -self.ce.c_eccentric
                * activation
                * f_isom
            )

            b_rel = (
                activation
                * f_isom
                * (1.0 - self.ce.c_eccentric)
                / (
                    activation * f_isom
                    + a_rel_con
                )
                * b_rel_con
                / self.ce.s_eccentric
            )

            d0 = (
                self.ce.l_ce_opt
                * b_rel
                * self.sde.d_se_max
                * (
                    self.sde.r_se
                    + (1.0 - self.sde.r_se)
                    * (
                        activation * f_isom
                        + f_pee / self.ce.f_max
                    )
                )
            )

            c2 = (
                self.sde.d_se_max
                * (
                    self.sde.r_se
                    - (
                        a_rel
                        - f_pee / self.ce.f_max
                    )
                    * (1.0 - self.sde.r_se)
                )
            )

            c1 = -(
                c2 * dot_l_mtc
                + d0
                + f_see
                - f_pee
                + self.ce.f_max * a_rel
            )

            c0 = (
                d0 * dot_l_mtc
                + self.ce.l_ce_opt
                * b_rel
                * (
                    f_see
                    - f_pee
                    - self.ce.f_max
                    * activation
                    * f_isom
                )
            )

            discriminant = (
                c1 * c1
                - 4.0 * c2 * c0
            )

            if discriminant < 0:
                discriminant = 0.0

            dot_l_ce = (
                -c1
                + math.sqrt(discriminant)
            ) / (2.0 * c2)

        f_ce = (
            self.ce.f_max
            * (
                (
                    activation * f_isom
                    + a_rel
                )
                /
                (
                    1.0
                    - (
                        dot_l_ce
                        /
                        (
                            b_rel
                            * self.ce.l_ce_opt
                        )
                    )
                )
                - a_rel
            )
        )

        f_sde = self.sde.get_force(
            f_ce=f_ce,
            f_pee=f_pee,
            f_max=self.ce.f_max,
            dot_l_mtc=dot_l_mtc,
            dot_l_ce=dot_l_ce,
        )

        force = f_see + f_sde

        return force, dot_l_ce

    def step(
        self,
        activation,
        l_mtc,
        dot_l_mtc,
        dt,
    ):
        force, dot_l_ce = self.compute(
            l_mtc=l_mtc,
            dot_l_mtc=dot_l_mtc,
            activation=activation,
        )

        self.l_ce_state += (
            dot_l_ce * dt
        )

        return force

def create_default_hill_pam(
    a_init=0.01,
    l_mtc_change_init=0.0,
):
    ce = ContractileElement(
        f_max=1500.0,
        l_ce_opt=0.2552835868056901,
        delta_w_limb_desc=0.46911026148677926,
        delta_w_limb_asc=0.3287538789247837,
        limb_desc=6.268913133974097,
        limb_asc=3.5061872661510742,
        a_rel0=0.24695507837001096,
        b_rel0=2.3725017725855437,
        s_eccentric=7.53956000448464,
        c_eccentric=1.185252544279298,
    )

    pee = ParallelElasticElement(
        contractile_element=ce,
        L=0.9292938812496556,
        v=2.625216390705555,
        F=17.5596788744868,
    )

    sde = SerialDampingElement(
        contractile_element=ce,
        d_se=0.9997764406809702,
        r_se=0.1,
    )

    see = SerialElasticElement(
        l_0=0.45879161390976636,
        delta_u_nll=0.042700142889960185,
        delta_u_l=0.01720377485875367,
        delta_f=1499.5759703926149,
    )

    return HillPAM(
        contractile_element=ce,
        parallel_elastic=pee,
        serial_damping=sde,
        serial_elastic=see,
        a_init=a_init,
        l_mtc_change_init=l_mtc_change_init,
        length=0.68,
    )

def create_hill_pam_from_json(
    path,
    a_init=0.01,
    l_mtc_change_init=0.0,
):
    with open(path, "r") as f:
        params = json.load(f)

    c = params["contractile"]
    p = params["parallel_elastic"]
    sd = params["serial_damping"]
    se = params["serial_elastic"]

    ce = ContractileElement(
        f_max=c["f_max"],
        l_ce_opt=c["l_CEopt"],
        delta_w_limb_desc=c["delta_w_limb_desc"],
        delta_w_limb_asc=c["delta_w_limb_asc"],
        limb_desc=c["limb_desc"],
        limb_asc=c["limb_asc"],
        a_rel0=c["a_rel0"],
        b_rel0=c["b_rel0"],
        s_eccentric=c["s_eccentric"],
        c_eccentric=c["c_eccentric"],
    )

    pee = ParallelElasticElement(
        ce,
        L=p["L"],
        v=p["v"],
        F=p["F"],
    )

    sde = SerialDampingElement(
        ce,
        d_se=sd["d_se"],
        r_se=sd["r_se"],
    )

    see = SerialElasticElement(
        l_0=se["l"],
        delta_u_nll=se["delta_u_nll"],
        delta_u_l=se["delta_u_l"],
        delta_f=se["delta_f"],
    )

    return HillPAM(
        contractile_element=ce,
        parallel_elastic=pee,
        serial_damping=sde,
        serial_elastic=see,
        a_init=a_init,
        l_mtc_change_init=l_mtc_change_init,
        length=params["length"],
    )
