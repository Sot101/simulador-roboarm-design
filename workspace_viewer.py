#!/usr/bin/env python3
"""
workspace_viewer.py — Visualizador 3D interactivo del *reachable workspace*
de un brazo robótico serial de 1 a 6 DOF (matplotlib standalone).

Uso:
    python workspace_viewer.py

Cinemática
----------
* Cadena serial de juntas revolutas con convención DH **estándar**:
      T_i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
* Cada preset puede llevar una transformación fija de base (p.ej. para que un
  péndulo oscile en el plano vertical XZ en lugar del plano horizontal XY).
* El gripper es un eslabón FIJO al final de la cadena: no es un DOF, no tiene
  slider ni rango; solo desplaza el punto de referencia del end-effector
  (TCP) a lo largo del eje x o z del último frame según el preset.

Cálculo del workspace
---------------------
* Monte Carlo: se muestrean N configuraciones articulares con distribución
  uniforme dentro del rango de cada junta, se evalúa la FK vectorizada
  (numpy, (N,4,4) matmul) y se acumulan las posiciones del TCP.
* N configurable con slider (500–20000, 4000 por defecto). El cálculo se
  dispara manualmente con el botón "Recalcular workspace"; mover los sliders
  de longitud solo actualiza el brazo y marca la nube como desactualizada.

Limitaciones del MVP (herramienta cualitativa de diseño, no análisis riguroso)
------------------------------------------------------------------------------
* NO considera auto-colisión ni colisión con el entorno/suelo.
* NO distingue *dexterous workspace* (alcanzable con cualquier orientación)
  de *reachable workspace* (alcanzable con al menos una orientación): solo se
  grafica la posición del TCP, la orientación se ignora.
* NO analiza singularidades ni manipulabilidad.
* La densidad de puntos NO es proporcional a "facilidad de alcance": es un
  artefacto del muestreo uniforme en espacio articular (mapeo no lineal).
* La nube es un muestreo finito: el borde real del workspace queda ligeramente
  fuera de la nube y regiones delgadas pueden aparecer vacías.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
N_SAMPLES_DEFAULT = 4000
N_SAMPLES_MIN, N_SAMPLES_MAX = 500, 20000
LEN_MIN, LEN_MAX = 0.0, 0.60          # rango de los sliders de longitud [m]
MAX_DOF = 6
LINK_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
GRIPPER_COLOR = "#333333"

# --- Estilo de la GUI (diseñada para caber en 1366x768 a 100 dpi) ----------
FIG_SIZE = (13.2, 7.0)
FIG_DPI = 100
PANEL_X0, PANEL_X1 = 0.655, 0.985     # columna derecha de controles
FS_TITLE, FS_HEADER, FS_BODY, FS_SMALL = 15, 11.5, 11, 9.5
C_BG = "#f7f7f9"          # fondo de la figura
C_PANEL = "#ffffff"       # fondo de widgets
C_ACCENT = "#2f6fed"      # azul de acción principal
C_ACCENT_HOVER = "#4f86f0"
C_STALE = "#e8743b"       # naranja: workspace desactualizado
C_STALE_HOVER = "#f08b58"
C_BTN = "#e9ecf1"
C_BTN_HOVER = "#d9dee7"
C_TEXT_MUTED = "#555a64"


# ---------------------------------------------------------------------------
# Utilidades de transformaciones homogéneas
# ---------------------------------------------------------------------------
def rotx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def trans(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def dh(theta, d, a, alpha):
    """Matriz DH estándar 4x4 (ángulos en radianes)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def dh_batch(theta, d, a, alpha):
    """Versión vectorizada: theta es (N,), devuelve (N,4,4)."""
    n = theta.shape[0]
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


# ---------------------------------------------------------------------------
# Presets por DOF
# ---------------------------------------------------------------------------
# Cada junta:  name, a [m], alpha [deg], d [m], rng (min,max) [deg],
#              len_key: qué parámetro DH ('a' o 'd') controla el slider de
#              longitud de ese eslabón.
# Cada preset: desc, base (T fija antes de J1), joints, home (pose por
#              defecto en grados), gripper (longitud [m]) y gripper_axis
#              ('x' o 'z' del último frame).
#
# Tabla resumen (longitudes en m, rangos en grados):
#   1 DOF  péndulo simple           J1 pitch  L=0.30  ±180
#   2 DOF  planar 2R (plano XZ)     J1 pitch  L=0.30  ±180 | J2 pitch L=0.25 ±150
#   3 DOF  antropomórfico           J1 yaw d=0.10 ±170 | J2 hombro a=0.30 ±90
#                                   J3 codo a=0.25 ±135
#   4 DOF  3 DOF + muñeca pitch     J4 pitch a=0.10 ±120
#   5 DOF  3 DOF + muñeca pitch/roll  J4 pitch a=0.00 ±120 | J5 roll d=0.08 ±180
#   6 DOF  yaw, hombro, codo(+offset), roll antebrazo, muñeca pitch, roll
#          J3 codo a=0.05 ±135 | J4 roll d=0.25 ±180 | J5 pitch a=0 ±120
#          J6 roll d=0.08 ±180
# ---------------------------------------------------------------------------
def J(name, a, alpha, d, rng, len_key):
    return dict(name=name, a=a, alpha=alpha, d=d, rng=rng, len_key=len_key)


# Base girada +90° en X: el eje z0 (eje de la junta) queda alineado con -Y del
# mundo, así el brazo planar se mueve en el plano vertical XZ (theta>0 = sube).
_PLANAR_BASE = rotx(90)

PRESETS = {
    1: dict(
        desc="Péndulo simple (1 pitch, plano XZ)",
        base=_PLANAR_BASE, gripper=0.06, gripper_axis="x",
        joints=[J("J1 pitch", 0.30, 0, 0.0, (-180, 180), "a")],
        home=[-35],
    ),
    2: dict(
        desc="Planar 2R (plano XZ)",
        base=_PLANAR_BASE, gripper=0.06, gripper_axis="x",
        joints=[
            J("J1 pitch", 0.30, 0, 0.0, (-180, 180), "a"),
            J("J2 pitch", 0.25, 0, 0.0, (-150, 150), "a"),
        ],
        home=[40, -70],
    ),
    3: dict(
        desc="Antropomórfico: yaw + hombro + codo",
        base=np.eye(4), gripper=0.06, gripper_axis="x",
        joints=[
            J("J1 yaw",    0.00, 90, 0.10, (-170, 170), "d"),
            J("J2 hombro", 0.30,  0, 0.00, (-90,   90), "a"),
            J("J3 codo",   0.25,  0, 0.00, (-135, 135), "a"),
        ],
        home=[30, 50, -80],
    ),
    4: dict(
        desc="Antropomórfico + muñeca pitch",
        base=np.eye(4), gripper=0.06, gripper_axis="x",
        joints=[
            J("J1 yaw",       0.00, 90, 0.10, (-170, 170), "d"),
            J("J2 hombro",    0.30,  0, 0.00, (-90,   90), "a"),
            J("J3 codo",      0.25,  0, 0.00, (-135, 135), "a"),
            J("J4 muñeca P",  0.10,  0, 0.00, (-120, 120), "a"),
        ],
        home=[30, 50, -80, -30],
    ),
    5: dict(
        desc="Antropomórfico + muñeca pitch/roll",
        base=np.eye(4), gripper=0.06, gripper_axis="z",
        joints=[
            J("J1 yaw",       0.00,  90, 0.10, (-170, 170), "d"),
            J("J2 hombro",    0.30,   0, 0.00, (-90,   90), "a"),
            J("J3 codo",      0.25,   0, 0.00, (-135, 135), "a"),
            J("J4 muñeca P",  0.00,  90, 0.00, (-120, 120), "a"),
            J("J5 muñeca R",  0.00,   0, 0.08, (-180, 180), "d"),
        ],
        home=[30, 50, -80, 60, 0],
    ),
    6: dict(
        desc="6R tipo PUMA: yaw, hombro, codo, roll, pitch, roll",
        base=np.eye(4), gripper=0.06, gripper_axis="z",
        joints=[
            J("J1 yaw",        0.00,  90, 0.10, (-170, 170), "d"),
            J("J2 hombro",     0.30,   0, 0.00, (-90,   90), "a"),
            J("J3 codo",       0.05,  90, 0.00, (-135, 135), "a"),
            J("J4 antebrazo R",0.00, -90, 0.25, (-180, 180), "d"),
            J("J5 muñeca P",   0.00,  90, 0.00, (-120, 120), "a"),
            J("J6 muñeca R",   0.00,   0, 0.08, (-180, 180), "d"),
        ],
        home=[30, 50, 10, 0, 60, 0],
    ),
}


# ---------------------------------------------------------------------------
# Modelo del brazo
# ---------------------------------------------------------------------------
class Arm:
    def __init__(self, dof):
        self.set_dof(dof)

    def set_dof(self, dof):
        p = PRESETS[dof]
        self.dof = dof
        self.desc = p["desc"]
        self.base = p["base"]
        self.joints = [dict(j) for j in p["joints"]]   # copia editable
        self.home = list(p["home"])
        self.gripper_len = p["gripper"]
        self.gripper_axis = p["gripper_axis"]

    # -- longitudes ---------------------------------------------------------
    def link_length(self, i):
        j = self.joints[i]
        return j[j["len_key"]]

    def set_link_length(self, i, value):
        j = self.joints[i]
        j[j["len_key"]] = float(value)

    def gripper_T(self):
        if self.gripper_axis == "x":
            return trans(x=self.gripper_len)
        return trans(z=self.gripper_len)

    def max_reach(self):
        """Cota superior conservadora del alcance (suma de |a| + |d| + gripper)."""
        return sum(abs(j["a"]) + abs(j["d"]) for j in self.joints) + self.gripper_len

    # -- FK -----------------------------------------------------------------
    def fk_chain(self, q_deg):
        """Devuelve (orígenes de frames (n+1,3), T_último, T_tcp)."""
        T = self.base.copy()
        pts = [T[:3, 3].copy()]
        for j, q in zip(self.joints, q_deg):
            T = T @ dh(np.radians(q), j["d"], j["a"], np.radians(j["alpha"]))
            pts.append(T[:3, 3].copy())
        return np.array(pts), T, T @ self.gripper_T()

    def sample_workspace(self, n, rng):
        """Monte Carlo: n muestras uniformes en espacio articular -> TCP (n,3)."""
        T = np.broadcast_to(self.base, (n, 4, 4)).copy()
        for j in self.joints:
            lo, hi = np.radians(j["rng"])
            th = rng.uniform(lo, hi, n)
            T = T @ dh_batch(th, j["d"], j["a"], np.radians(j["alpha"]))
        T = T @ self.gripper_T()
        return T[:, :3, 3]

    def random_pose(self, rng):
        return [rng.uniform(*j["rng"]) for j in self.joints]


# ---------------------------------------------------------------------------
# Aplicación (figura + widgets)
# ---------------------------------------------------------------------------
class WorkspaceApp:
    def __init__(self, dof=3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.arm = Arm(dof)
        self.pose = list(self.arm.home)
        self.cloud = None          # np.ndarray (N,3) o None
        self.cloud_stale = True
        self.scatter = None

        self._apply_style()
        self._build_figure()
        self._build_panel()
        self._sync_sliders_to_arm()
        self.recalc()

    # -- estilo global ------------------------------------------------------
    @staticmethod
    def _apply_style():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": FS_BODY,
            "axes.labelsize": FS_BODY,
            "xtick.labelsize": FS_SMALL,
            "ytick.labelsize": FS_SMALL,
            "figure.facecolor": C_BG,
            "toolbar": "none",       # sin barra de herramientas: más área útil
        })

    # -- helpers de layout --------------------------------------------------
    def _header(self, y, text):
        """Cabecera de sección: texto en negrita + línea separadora."""
        self.fig.text(PANEL_X0, y, text, fontsize=FS_HEADER, weight="bold",
                      va="bottom", color="#222")
        self.fig.add_artist(plt.Line2D([PANEL_X0, PANEL_X1], [y - 0.006] * 2,
                                       color="#c9ccd3", lw=1.2,
                                       transform=self.fig.transFigure))

    def _button(self, rect, text, primary=False, fs=FS_BODY, bold=False):
        ax = self.fig.add_axes(rect)
        b = Button(ax, text,
                   color=C_ACCENT if primary else C_BTN,
                   hovercolor=C_ACCENT_HOVER if primary else C_BTN_HOVER)
        b.label.set_fontsize(fs)
        b.label.set_color("white" if primary else "#222")
        if bold:
            b.label.set_weight("bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#b8bcc6")
        return b

    def _slider(self, rect, label, vmin, vmax, vinit, step, fmt, color):
        ax = self.fig.add_axes(rect, facecolor=C_PANEL)
        s = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step,
                   valfmt=fmt, color=color, track_color="#e1e4ea",
                   handle_style=dict(size=13, facecolor="white",
                                     edgecolor=color))
        s.label.set_fontsize(FS_BODY)
        s.label.set_horizontalalignment("right")
        s.label.set_position((-0.03, 0.5))
        s.valtext.set_fontsize(FS_BODY)
        s.valtext.set_position((1.03, 0.5))
        for sp in ax.spines.values():
            sp.set_visible(False)
        return s

    # -- construcción -------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        try:
            self.fig.canvas.manager.set_window_title(
                "Reachable workspace — brazo serial 1–6 DOF")
        except Exception:
            pass

        # --- Vista 3D ---
        self.ax = self.fig.add_axes([0.0, 0.09, 0.645, 0.89], projection="3d")
        self.ax.set_xlabel("X [m]", labelpad=8)
        self.ax.set_ylabel("Y [m]", labelpad=8)
        self.ax.set_zlabel("Z [m]", labelpad=6)
        self.ax.view_init(elev=24, azim=-58)
        self.ax.set_facecolor(C_BG)
        self.reach_text = self.fig.text(0.02, 0.955, "", fontsize=FS_BODY,
                                        color=C_TEXT_MUTED, va="top")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0.9))
            axis._axinfo["grid"].update(color="#d5d8de", linewidth=0.7)

        # Suelo: círculo de alcance en z=0 + cruz en el origen
        self.ground_circle = self.ax.plot([], [], [], "--", lw=1.1,
                                          color="#9aa0aa")[0]
        self.ground_cross = [self.ax.plot([], [], [], "-", lw=0.9,
                                          color="#b5bac3")[0] for _ in range(2)]

        # Artistas del brazo (se crean una vez, luego set_data_3d)
        self.link_lines = [
            self.ax.plot([], [], [], "-", lw=5, color=LINK_COLORS[i],
                         solid_capstyle="round")[0]
            for i in range(MAX_DOF)
        ]
        self.joint_dots = self.ax.plot([], [], [], "o", ms=8, color="#222",
                                       mec="white", mew=1.2, zorder=10)[0]
        self.base_dot = self.ax.plot([0], [0], [0], "s", ms=11, color="#222")[0]
        self.grip_line = self.ax.plot([], [], [], "-", lw=3.5,
                                      color=GRIPPER_COLOR)[0]
        self.finger_lines = [self.ax.plot([], [], [], "-", lw=2.5,
                                          color=GRIPPER_COLOR)[0]
                             for _ in range(3)]
        self.tcp_dot = self.ax.plot([], [], [], "*", ms=15, color="gold",
                                    mec="#222", zorder=11)[0]

        # Colorbar horizontal (altura Z del TCP) bajo la vista 3D
        self.norm = plt.Normalize(-1, 1)
        self.cmap = plt.get_cmap("viridis")
        cax = self.fig.add_axes([0.11, 0.065, 0.40, 0.022])
        sm = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.cbar = self.fig.colorbar(sm, cax=cax, orientation="horizontal")
        self.cbar.set_label("Altura Z del TCP [m]", fontsize=FS_SMALL + 0.5)
        cax.tick_params(labelsize=FS_SMALL)

    def _build_panel(self):
        X0, X1 = PANEL_X0, PANEL_X1
        W = X1 - X0

        # --- Título ---
        self.fig.text(X0, 0.962, "Reachable workspace · Monte Carlo",
                      fontsize=FS_TITLE, weight="bold", va="bottom")
        self.desc_text = self.fig.text(X0, 0.928, "", fontsize=FS_BODY,
                                       style="italic", color=C_TEXT_MUTED,
                                       va="bottom")

        # =============== Sección 1: configuración =========================
        self._header(0.885, "Configuración")

        ax_radio = self.fig.add_axes([X0, 0.615, 0.10, 0.255],
                                     facecolor=C_PANEL)
        self.radio = RadioButtons(
            ax_radio, [f"{i} DOF" for i in range(1, MAX_DOF + 1)],
            active=self.arm.dof - 1,
            label_props=dict(fontsize=[FS_BODY] * MAX_DOF),
            radio_props=dict(s=[90] * MAX_DOF, facecolor=[C_ACCENT] * MAX_DOF,
                             edgecolor=["#666"] * MAX_DOF),
        )
        self.radio.on_clicked(self.on_dof)

        ax_chk = self.fig.add_axes([0.77, 0.775, X1 - 0.77, 0.095],
                                   facecolor=C_PANEL)
        self.chk = CheckButtons(
            ax_chk, ["Brazo (pose actual)", "Nube del workspace"],
            [True, True],
            label_props=dict(fontsize=[FS_BODY] * 2),
            frame_props=dict(s=[150] * 2, edgecolor=["#666"] * 2),
            check_props=dict(s=[150] * 2, facecolor=[C_ACCENT] * 2),
        )
        self.chk.on_clicked(self.on_toggle)

        self.btn_home = self._button([0.77, 0.70, 0.10, 0.052], "Pose home")
        self.btn_home.on_clicked(self.on_home_pose)
        self.btn_pose = self._button([0.88, 0.70, X1 - 0.88, 0.052],
                                     "Pose aleatoria")
        self.btn_pose.on_clicked(self.on_random_pose)

        # =============== Sección 2: longitudes ============================
        self._header(0.585, "Longitudes de eslabón  [m]")
        self.btn_reset = self._button([0.865, 0.586, X1 - 0.865, 0.04],
                                      "Reset", fs=FS_SMALL + 0.5)
        self.btn_reset.on_clicked(self.on_reset_lengths)

        self.sliders = []
        sx0, sw = 0.735, 0.185
        y0, dy = 0.523, 0.046
        for i in range(MAX_DOF):
            s = self._slider([sx0, y0 - i * dy, sw, 0.03], f"L{i + 1}",
                             LEN_MIN, LEN_MAX, 0.1, 0.005, "%.3f",
                             LINK_COLORS[i])
            s.on_changed(lambda val, idx=i: self.on_length(idx, val))
            self.sliders.append(s)

        # =============== Sección 3: workspace =============================
        self._header(0.245, "Workspace  (Monte Carlo)")
        self.s_n = self._slider([sx0, 0.19, sw, 0.03], "N muestras",
                                N_SAMPLES_MIN, N_SAMPLES_MAX, N_SAMPLES_DEFAULT,
                                500, "%d", "#6c757d")
        self.btn_recalc = self._button([X0, 0.112, W, 0.06],
                                       "Recalcular workspace",
                                       primary=True, fs=FS_BODY + 1.5, bold=True)
        self.btn_recalc.on_clicked(lambda _e: self.recalc())

        # --- Estado ---
        self.status = self.fig.text(
            X0 + 0.008, 0.096, "", fontsize=FS_SMALL, family="DejaVu Sans Mono",
            va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=C_PANEL,
                      edgecolor="#c9ccd3"))

    # -- sincronización sliders <-> modelo ---------------------------------
    def _sync_sliders_to_arm(self):
        """Muestra solo los sliders del DOF actual y carga sus valores."""
        for i, s in enumerate(self.sliders):
            active = i < self.arm.dof
            s.ax.set_visible(active)
            if active:
                j = self.arm.joints[i]
                s.label.set_text(f"L{i + 1}  ({j['len_key']}{i + 1})")
                s.eventson = False
                s.set_val(self.arm.link_length(i))
                s.eventson = True
        self.desc_text.set_text(f"{self.arm.dof} DOF — {self.arm.desc}")

    def _set_recalc_style(self, stale):
        col, hov = (C_STALE, C_STALE_HOVER) if stale else (C_ACCENT, C_ACCENT_HOVER)
        self.btn_recalc.color, self.btn_recalc.hovercolor = col, hov
        self.btn_recalc.ax.set_facecolor(col)
        self.btn_recalc.label.set_text(
            "Recalcular workspace  (desactualizado)" if stale
            else "Recalcular workspace")

    # -- callbacks ----------------------------------------------------------
    def on_dof(self, label):
        self.arm.set_dof(int(label.split()[0]))
        self.pose = list(self.arm.home)
        self._sync_sliders_to_arm()
        self._mark_stale()
        self.draw_arm()
        self.update_limits()
        self.fig.canvas.draw_idle()

    def on_length(self, idx, val):
        if idx >= self.arm.dof:
            return
        self.arm.set_link_length(idx, val)
        self._mark_stale()
        self.draw_arm()
        self.update_limits()
        self.fig.canvas.draw_idle()

    def on_toggle(self, _label):
        self._apply_visibility()
        self.fig.canvas.draw_idle()

    def on_random_pose(self, _e):
        self.pose = self.arm.random_pose(self.rng)
        self.draw_arm()
        self.fig.canvas.draw_idle()

    def on_home_pose(self, _e):
        self.pose = list(self.arm.home)
        self.draw_arm()
        self.fig.canvas.draw_idle()

    def on_reset_lengths(self, _e):
        preset = PRESETS[self.arm.dof]["joints"]
        for i, pj in enumerate(preset):
            self.arm.joints[i][pj["len_key"]] = pj[pj["len_key"]]
        self._sync_sliders_to_arm()
        self._mark_stale()
        self.draw_arm()
        self.update_limits()
        self.fig.canvas.draw_idle()

    # -- cálculo del workspace ---------------------------------------------
    def recalc(self):
        n = int(self.s_n.val)
        t0 = time.perf_counter()
        self.cloud = self.arm.sample_workspace(n, self.rng)
        dt = time.perf_counter() - t0
        self.cloud_stale = False
        self._set_recalc_style(stale=False)

        r = np.linalg.norm(self.cloud, axis=1)
        mins, maxs = self.cloud.min(0), self.cloud.max(0)
        self.status.set_text(
            f"N = {n}   FK {dt * 1000:.0f} ms   "
            f"r_max {r.max():.3f} m  (cota {self.arm.max_reach():.3f})\n"
            f"X [{mins[0]:+.2f}, {maxs[0]:+.2f}]  "
            f"Y [{mins[1]:+.2f}, {maxs[1]:+.2f}]  "
            f"Z [{mins[2]:+.2f}, {maxs[2]:+.2f}]"
        )
        self.status.set_color("#222")
        self.draw_cloud()
        self.draw_arm()
        self.update_limits()
        self.fig.canvas.draw_idle()

    def _mark_stale(self):
        self.cloud_stale = True
        self._set_recalc_style(stale=True)
        if self.scatter is not None:
            self.scatter.set_alpha(0.08)   # nube atenuada = desactualizada
        self.status.set_text("La nube mostrada NO corresponde a la geometría "
                             "actual.\nPulsa «Recalcular workspace».")
        self.status.set_color(C_STALE)

    # -- dibujo -------------------------------------------------------------
    def draw_cloud(self):
        if self.scatter is not None:
            self.scatter.remove()
            self.scatter = None
        if self.cloud is None:
            return
        P = self.cloud
        zmin, zmax = P[:, 2].min(), P[:, 2].max()
        if zmax - zmin < 1e-9:
            zmin, zmax = zmin - 0.01, zmax + 0.01
        self.norm.vmin, self.norm.vmax = zmin, zmax
        self.cbar.update_normal(self.cbar.mappable)
        self.scatter = self.ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                                       c=P[:, 2], cmap=self.cmap, norm=self.norm,
                                       s=5, alpha=0.55, linewidths=0,
                                       depthshade=False)
        self._apply_visibility()

    def draw_arm(self):
        pts, T_last, T_tcp = self.arm.fk_chain(self.pose)
        n = self.arm.dof
        for i, line in enumerate(self.link_lines):
            if i < n:
                seg = pts[i:i + 2]
                line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
            else:
                line.set_data_3d([], [], [])
        self.joint_dots.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

        # Gripper: barra desde el último frame hasta el TCP + dos dedos
        wrist, tcp = pts[-1], T_tcp[:3, 3]
        self.grip_line.set_data_3d([wrist[0], tcp[0]], [wrist[1], tcp[1]],
                                   [wrist[2], tcp[2]])
        R = T_last[:3, :3]
        fwd = R[:, 0] if self.arm.gripper_axis == "x" else R[:, 2]
        perp = R[:, 1] if self.arm.gripper_axis == "x" else R[:, 0]
        g = self.arm.gripper_len
        hw, fl = 0.30 * g, 0.45 * g       # semiancho y largo de dedos
        palm = tcp - fwd * fl
        a, b = palm + perp * hw, palm - perp * hw
        segs = [(a, b), (a, a + fwd * fl), (b, b + fwd * fl)]
        for line, (p, q) in zip(self.finger_lines, segs):
            line.set_data_3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])
        self.tcp_dot.set_data_3d([tcp[0]], [tcp[1]], [tcp[2]])
        self._apply_visibility()

    def _apply_visibility(self):
        show_arm, show_cloud = self.chk.get_status()
        for art in (*self.link_lines, self.joint_dots, self.grip_line,
                    *self.finger_lines, self.tcp_dot):
            art.set_visible(show_arm)
        if self.scatter is not None:
            self.scatter.set_visible(show_cloud)

    def update_limits(self):
        """Límites de ejes = alcance máximo del brazo (cúbicos, aspecto 1:1:1)."""
        reach = self.arm.max_reach()
        R = max(reach * 1.05, 0.05)
        self.ax.set_xlim(-R, R)
        self.ax.set_ylim(-R, R)
        self.ax.set_zlim(-R, R)
        self.ax.set_box_aspect((1, 1, 1))
        self.reach_text.set_text(f"Alcance máximo teórico: {reach:.3f} m")
        # suelo
        t = np.linspace(0, 2 * np.pi, 121)
        self.ground_circle.set_data_3d(reach * np.cos(t), reach * np.sin(t),
                                       np.zeros_like(t))
        self.ground_cross[0].set_data_3d([-reach, reach], [0, 0], [0, 0])
        self.ground_cross[1].set_data_3d([0, 0], [-reach, reach], [0, 0])


def main():
    app = WorkspaceApp(dof=3)
    plt.show()
    return app


if __name__ == "__main__":
    main()
