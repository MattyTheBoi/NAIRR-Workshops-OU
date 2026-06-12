"""Plotting and animation helpers for N2 — Kalman Filter from Scratch.

All visualization code lives in this module so the notebook cells stay
focused on the filter math. Nothing in here changes the algorithms —
every function just draws.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


# ---------------------------------------------------------------- helpers ---

def _show_animation(fig, update, n_frames, interval=60):
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    print(f"Animation: {n_frames} frames  |  Press ▶ to play")
    display(HTML(anim.to_jshtml()))


def _ellipse_params(cov_xy, n_sigma=2):
    """Width, height, angle of the n-sigma confidence ellipse for a 2x2 covariance."""
    evals, evecs = np.linalg.eigh(cov_xy)
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    w, h = 2 * n_sigma * np.sqrt(evals)
    return w, h, angle


# --------------------------------------------------------- 1D tracking ------

def animate_1d(true_positions, measurements, hist, true_velocity):
    """Animated position tracking (top) and velocity estimate (bottom) with 2-sigma bands."""
    x_est, P_est = hist["x"], hist["P"]
    num_steps = len(measurements)
    time = np.arange(num_steps)
    pos_std = np.sqrt(P_est[:, 0, 0])
    vel_std = np.sqrt(P_est[:, 1, 1])

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(14, 9), dpi=80,
                                         gridspec_kw={"height_ratios": [3, 1]})

    ax_pos.set_xlabel("Time step", fontsize=12)
    ax_pos.set_ylabel("Position (m)", fontsize=12)
    ax_pos.set_title("1D Kalman Filter — Constant-Velocity Tracking", fontsize=14)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_xlim(-1, num_steps)
    ax_pos.set_ylim(min(measurements.min(), true_positions.min()) - 15,
                    max(measurements.max(), true_positions.max()) + 15)

    true_ln, = ax_pos.plot([], [], "k-", linewidth=2, label="True position")
    meas_sc = ax_pos.scatter([], [], c="red", s=15, alpha=0.4,
                             label="Noisy measurements", zorder=3)
    est_ln, = ax_pos.plot([], [], "b-", linewidth=2, label="Kalman estimate")
    ax_pos.legend(fontsize=11, loc="upper left")

    ax_vel.axhline(true_velocity, color="k", linewidth=1.5, linestyle="--",
                   label="True velocity")
    ax_vel.set_xlabel("Time step", fontsize=12)
    ax_vel.set_ylabel("Velocity (m/s)", fontsize=12)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_xlim(-1, num_steps)
    ax_vel.set_ylim(min(x_est[:, 1].min() - 2, -1),
                    max(true_velocity + 2, x_est[:, 1].max() + 2))

    vel_ln, = ax_vel.plot([], [], "b-", linewidth=2, label="Estimated velocity")
    ax_vel.legend(fontsize=11)
    plt.tight_layout()

    bands = [None, None]

    def update(frame):
        s = slice(0, frame + 1)
        t = time[s]

        true_ln.set_data(t, true_positions[s])
        meas_sc.set_offsets(np.column_stack([t, measurements[s]]))
        est_ln.set_data(t, x_est[s, 0])
        vel_ln.set_data(t, x_est[s, 1])

        for i, (ax, mean, std) in enumerate([(ax_pos, x_est[s, 0], pos_std[s]),
                                             (ax_vel, x_est[s, 1], vel_std[s])]):
            if bands[i] is not None:
                bands[i].remove()
            bands[i] = ax.fill_between(t, mean - 2 * std, mean + 2 * std,
                                       color="blue", alpha=0.15)

        return (true_ln, meas_sc, est_ln, vel_ln)

    _show_animation(fig, update, num_steps)


# -------------------------------------------------- gain convergence --------

def plot_gain_convergence(hist):
    """Static plot: Kalman gain converging (left), predict-widens/update-narrows (right)."""
    K, P_est, P_pred = hist["K"], hist["P"], hist["P_pred"]
    time = np.arange(len(K))
    pred_std = np.sqrt(P_pred[:, 0, 0])
    post_std = np.sqrt(P_est[:, 0, 0])

    fig, (ax_gain, ax_unc) = plt.subplots(1, 2, figsize=(16, 5), dpi=80)

    ax_gain.plot(time, K[:, 0, 0], "b-", linewidth=2, label="K (position)")
    ax_gain.plot(time, K[:, 1, 0], "r-", linewidth=2, label="K (velocity)")
    ax_gain.set_xlabel("Time step", fontsize=12)
    ax_gain.set_ylabel("Kalman Gain", fontsize=12)
    ax_gain.set_title("Kalman Gain Convergence", fontsize=14)
    ax_gain.grid(True, alpha=0.3)
    ax_gain.legend(fontsize=11)

    ax_unc.plot(time, pred_std, "r--", linewidth=2, label="Predicted σ (before update)")
    ax_unc.plot(time, post_std, "b-", linewidth=2, label="Posterior σ (after update)")
    ax_unc.set_xlabel("Time step", fontsize=12)
    ax_unc.set_ylabel("Position Std Dev (m)", fontsize=12)
    ax_unc.set_title("Uncertainty: Predict Widens, Update Narrows", fontsize=14)
    ax_unc.grid(True, alpha=0.3)
    ax_unc.legend(fontsize=11)

    plt.tight_layout()
    plt.show()
    print(f"Gain converges: {K[0, 0, 0]:.3f} → {K[-1, 0, 0]:.3f}")


# ------------------------------------------------------- Q/R tuning ---------

def plot_qr_comparison(true_pos, measurements, results):
    """Static stacked comparison of filter configs.

    results: list of (label, x_est, pos_std, rmse) tuples.
    """
    time = np.arange(len(true_pos))
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 4 * len(results)),
                             dpi=80, sharex=True)

    for ax, (label, x_est, pos_std, rmse) in zip(axes, results):
        ax.plot(time, true_pos, "k-", linewidth=2, label="True")
        ax.scatter(time, measurements, c="red", s=10, alpha=0.3, label="Measurements")
        ax.plot(time, x_est[:, 0], "b-", linewidth=2, label="Estimate")
        ax.fill_between(time, x_est[:, 0] - 2 * pos_std, x_est[:, 0] + 2 * pos_std,
                        color="blue", alpha=0.12)
        ax.set_ylabel("Position (m)", fontsize=11)
        ax.set_title(f"{label}  —  RMSE = {rmse:.2f} m", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper left")

    axes[-1].set_xlabel("Time step", fontsize=12)
    plt.tight_layout()
    plt.show()


def interactive_qr_explorer(kalman_filter, F, H, true_pos, measurements,
                            dt=1.0, true_noise_std=8.0):
    """Slider-driven Q/R explorer. The measurements stay fixed; the sliders
    only change the filter's assumed Q and R, so you can see what happens
    when the filter's noise model matches reality vs. when it's mis-specified.

    kalman_filter: the filter function defined in the notebook.
    """
    import ipywidgets as widgets

    num_steps = len(measurements)
    time = np.arange(num_steps)
    true_vel = np.gradient(true_pos, dt)

    # Fixed axis limits so the plots never rescale as you drag
    pos_ymin = min(true_pos.min(), measurements.min()) - 15
    pos_ymax = max(true_pos.max(), measurements.max()) + 15

    def run_and_plot(q_scale, r_scale):
        Q = np.array([[dt**4 / 4, dt**3 / 2],
                      [dt**3 / 2, dt**2]]) * q_scale**2
        R = np.array([[r_scale**2]])

        hist = kalman_filter(measurements.reshape(-1, 1), F, H, Q, R,
                             np.array([0.0, 0.0]), np.diag([500.0, 50.0]))
        x_est, P_est, K = hist["x"], hist["P"], hist["K"]
        pos_std = np.sqrt(P_est[:, 0, 0])
        vel_std = np.sqrt(P_est[:, 1, 1])
        rmse = np.sqrt(np.mean((x_est[:, 0] - true_pos)**2))
        qr_ratio = q_scale**2 / r_scale**2

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))

        ax = axes[0, 0]
        ax.plot(time, true_pos, 'k-', linewidth=2, label='True')
        ax.scatter(time, measurements, c='red', s=10, alpha=0.3,
                   label=f'Measurements (true noise = {true_noise_std})')
        ax.plot(time, x_est[:, 0], 'b-', linewidth=2, label='Kalman estimate')
        ax.fill_between(time, x_est[:, 0] - 2 * pos_std, x_est[:, 0] + 2 * pos_std,
                        color='blue', alpha=0.12)
        ax.set_xlim(-1, num_steps)
        ax.set_ylim(pos_ymin, pos_ymax)
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title(f'Position Tracking  |  RMSE = {rmse:.2f} m', fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(time, true_vel, 'k-', linewidth=2, label='True velocity')
        ax.plot(time, x_est[:, 1], 'b-', linewidth=2, label='Estimated velocity')
        ax.fill_between(time, x_est[:, 1] - 2 * vel_std, x_est[:, 1] + 2 * vel_std,
                        color='blue', alpha=0.12)
        ax.set_xlim(-1, num_steps)
        ax.set_ylim(true_vel.min() - 3, true_vel.max() + 3)
        ax.set_ylabel('Velocity (m/s)', fontsize=11)
        ax.set_title('Velocity Estimate', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(time, K[:, 0, 0], 'b-', linewidth=2, label='K (position)')
        ax.plot(time, K[:, 1, 0], 'r-', linewidth=2, label='K (velocity)')
        ax.set_xlim(-1, num_steps)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Time step', fontsize=11)
        ax.set_ylabel('Kalman Gain', fontsize=11)
        ax.set_title(f'Kalman Gain  |  steady-state K_pos = {K[-1, 0, 0]:.3f}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Summary panel: classify the filter's "personality" from the Q/R ratio
        ax = axes[1, 1]
        ax.axis('off')
        if qr_ratio < 0.005:
            personality, bg = "STIFF  (trusts model heavily)", '#d4e6f1'
        elif qr_ratio > 0.1:
            personality, bg = "FLOPPY  (trusts sensor heavily)", '#fadbd8'
        else:
            personality, bg = "BALANCED", '#d5f5e3'

        if abs(r_scale - true_noise_std) / true_noise_std < 0.15:
            r_status = "R ~ true noise  (well-calibrated)"
        elif r_scale > true_noise_std:
            r_status = "R > true noise  (over-smoothing)"
        else:
            r_status = "R < true noise  (over-trusting sensor)"

        info = (
            f"Filter's Q   sigma_q = {q_scale:.4f}\n"
            f"Filter's R   sigma_r = {r_scale:.2f}\n"
            f"True noise   sigma   = {true_noise_std:.2f}\n\n"
            f"Q/R ratio = {qr_ratio:.6f}\n\n"
            f"Position RMSE = {rmse:.2f} m\n"
            f"Noise reduction = {true_noise_std / max(rmse, 0.01):.1f}x\n\n"
            f"Steady-state gains:\n"
            f"  K_pos = {K[-1, 0, 0]:.4f}\n"
            f"  K_vel = {K[-1, 1, 0]:.4f}\n\n"
            f"R calibration: {r_status}\n"
            f"Personality:   {personality}"
        )
        ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bg, alpha=0.8))
        ax.set_title('Filter Summary', fontsize=13)

        plt.tight_layout()
        plt.show()

    q_slider = widgets.FloatLogSlider(
        value=0.5, base=10, min=-2, max=1.5, step=0.05,
        description='Q (process noise):',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='550px'),
        readout_format='.4f')

    r_slider = widgets.FloatLogSlider(
        value=8.0, base=10, min=0.3, max=1.7, step=0.05,
        description='R (meas. noise):',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='550px'),
        readout_format='.2f')

    out = widgets.interactive_output(run_and_plot,
                                     {'q_scale': q_slider, 'r_scale': r_slider})

    display(widgets.VBox([
        widgets.HTML(
            "<h4>Interactive Q / R Tuning</h4>"
            "<p style='color: gray; margin-top: -8px;'>"
            f"The measurements are fixed (true sensor noise = {true_noise_std:.1f}). "
            "The sliders only change the filter's Q and R parameters — so you can "
            "explore what happens when R matches reality vs. when it's mis-specified.</p>"
        ),
        q_slider, r_slider, out
    ]))


# ------------------------------------------------------- 2D tracking --------

def animate_2d(t, true_x, true_y, meas_x, meas_y, estimates, covariances, step=2):
    """Animated figure-eight tracking with confidence ellipse (left) and
    measurement-vs-estimate error over time (right)."""
    num_steps = len(t)
    pos_error = np.sqrt((estimates[:, 0] - true_x)**2 + (estimates[:, 1] - true_y)**2)
    meas_error = np.sqrt((meas_x - true_x)**2 + (meas_y - true_y)**2)
    pos_sigma = np.sqrt(covariances[:, 0, 0] + covariances[:, 1, 1])
    frame_indices = list(range(0, num_steps, step))

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7), dpi=80)

    ax_traj.plot(true_x, true_y, "k-", linewidth=1, alpha=0.15)
    ax_traj.set_xlabel("X (m)", fontsize=12)
    ax_traj.set_ylabel("Y (m)", fontsize=12)
    ax_traj.set_title("2D Kalman Filter — Figure-Eight Tracking", fontsize=14)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    margin = 12
    ax_traj.set_xlim(true_x.min() - margin, true_x.max() + margin)
    ax_traj.set_ylim(true_y.min() - margin, true_y.max() + margin)

    meas_scat = ax_traj.scatter([], [], c="red", s=12, alpha=0.3, label="Measurements")
    true_trail, = ax_traj.plot([], [], "k-", linewidth=2, label="True trajectory")
    est_trail, = ax_traj.plot([], [], "b-", linewidth=2, label="Kalman estimate")
    true_dot, = ax_traj.plot([], [], "ko", markersize=10, zorder=5)
    est_dot, = ax_traj.plot([], [], "bs", markersize=8, zorder=5)

    ell = Ellipse((0, 0), 1, 1, fill=False, edgecolor="blue", linewidth=1.5, alpha=0.6)
    ax_traj.add_patch(ell)
    ell.set_visible(False)
    ax_traj.legend(fontsize=10, loc="upper left")

    ax_err.set_xlabel("Time (s)", fontsize=12)
    ax_err.set_ylabel("Position Error (m)", fontsize=12)
    ax_err.set_title("Error: Measurements vs Kalman Estimate", fontsize=14)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(t[0], t[-1])
    ax_err.set_ylim(0, max(meas_error.max(), pos_error.max()) * 1.1)

    meas_err_line, = ax_err.plot([], [], "r-", alpha=0.3, linewidth=1, label="Measurement error")
    est_err_line, = ax_err.plot([], [], "b-", linewidth=2, label="Estimate error")
    sigma_line, = ax_err.plot([], [], "b--", linewidth=1.5, alpha=0.5, label="2σ bound")
    ax_err.legend(fontsize=11)
    plt.tight_layout()

    def update(frame):
        k = frame_indices[frame]
        s = slice(0, k + 1)

        meas_scat.set_offsets(np.column_stack([meas_x[s], meas_y[s]]))
        true_trail.set_data(true_x[s], true_y[s])
        est_trail.set_data(estimates[s, 0], estimates[s, 1])
        true_dot.set_data([true_x[k]], [true_y[k]])
        est_dot.set_data([estimates[k, 0]], [estimates[k, 1]])

        w, h, angle = _ellipse_params(covariances[k, :2, :2])
        ell.set_center((estimates[k, 0], estimates[k, 1]))
        ell.width, ell.height, ell.angle = w, h, angle
        ell.set_visible(True)

        meas_err_line.set_data(t[s], meas_error[s])
        est_err_line.set_data(t[s], pos_error[s])
        sigma_line.set_data(t[s], 2 * pos_sigma[s])

        return (meas_scat, true_trail, est_trail, true_dot, est_dot,
                ell, meas_err_line, est_err_line, sigma_line)

    _show_animation(fig, update, len(frame_indices))


# ----------------------------------------------------- sensor dropout -------

def animate_dropout(t, true_x, true_y, meas_x, meas_y, estimates, covariances,
                    dropout_start, dropout_end, step=2):
    """Animated dropout demo: trajectory phases colored tracking/coasting/recovery
    (left), uncertainty and error over time (right)."""
    num_steps = len(t)
    err = np.sqrt((estimates[:, 0] - true_x)**2 + (estimates[:, 1] - true_y)**2)
    pos_sigma = np.sqrt(covariances[:, 0, 0] + covariances[:, 1, 1])
    frame_indices = list(range(0, num_steps, step))

    fig, (ax_traj, ax_sig) = plt.subplots(1, 2, figsize=(16, 7), dpi=80)

    ax_traj.plot(true_x, true_y, "k-", linewidth=1, alpha=0.15)
    ax_traj.set_xlabel("X (m)", fontsize=12)
    ax_traj.set_ylabel("Y (m)", fontsize=12)
    ax_traj.set_title(f"Sensor Dropout — {dropout_end - dropout_start}-Step Blackout",
                      fontsize=14)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    all_x = np.concatenate([true_x, estimates[:, 0], meas_x])
    all_y = np.concatenate([true_y, estimates[:, 1], meas_y])
    pad = 10
    ax_traj.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_traj.set_ylim(all_y.min() - pad, all_y.max() + pad)

    meas_scat = ax_traj.scatter([], [], c="red", s=12, alpha=0.3, label="Measurements")
    true_trail, = ax_traj.plot([], [], "k-", linewidth=2, label="True trajectory")
    pre_line, = ax_traj.plot([], [], "b-", linewidth=2, label="Tracking")
    coast_line, = ax_traj.plot([], [], color="orange", linewidth=2.5, linestyle="--",
                               label="Coasting (no measurements)")
    post_line, = ax_traj.plot([], [], "g-", linewidth=2, label="Recovery")
    true_dot, = ax_traj.plot([], [], "ko", markersize=10, zorder=5)
    est_dot, = ax_traj.plot([], [], "s", color="blue", markersize=8, zorder=5)

    ell = Ellipse((0, 0), 1, 1, fill=False, edgecolor="blue", linewidth=1.5, alpha=0.6)
    ax_traj.add_patch(ell)
    ell.set_visible(False)

    status_text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes,
                               fontsize=12, fontweight="bold", va="top",
                               bbox=dict(boxstyle="round,pad=0.3",
                                         facecolor="white", alpha=0.8))
    ax_traj.legend(fontsize=9, loc="upper right")

    ax_sig.set_xlabel("Time (s)", fontsize=12)
    ax_sig.set_ylabel("(m)", fontsize=12)
    ax_sig.set_title("Uncertainty & Error Over Time", fontsize=14)
    ax_sig.grid(True, alpha=0.3)
    ax_sig.set_xlim(t[0], t[-1])
    ax_sig.set_ylim(0, max(pos_sigma.max(), err.max()) * 1.1)
    ax_sig.axvspan(t[dropout_start], t[dropout_end - 1],
                   color="orange", alpha=0.12, label="Dropout window")

    sig_trail, = ax_sig.plot([], [], "b-", linewidth=2, label="Position σ")
    err_trail, = ax_sig.plot([], [], "r-", linewidth=1.5, alpha=0.6, label="Position error")
    time_marker = ax_sig.axvline(0, color="gray", linewidth=1, alpha=0.4, linestyle="--")
    ax_sig.legend(fontsize=10)
    plt.tight_layout()

    def update(frame):
        k = frame_indices[frame]
        s = slice(0, k + 1)

        true_trail.set_data(true_x[s], true_y[s])
        true_dot.set_data([true_x[k]], [true_y[k]])

        # Measurements are only drawn outside the dropout window
        vis = np.arange(k + 1)
        vis = vis[(vis < dropout_start) | (vis >= dropout_end)]
        meas_scat.set_offsets(np.column_stack([meas_x[vis], meas_y[vis]])
                              if len(vis) else np.empty((0, 2)))

        # Estimate trail, split into tracking / coasting / recovery phases
        pre_end = min(k + 1, dropout_start)
        pre_line.set_data(estimates[:pre_end, 0], estimates[:pre_end, 1])

        if k >= dropout_start:
            coast_end = min(k + 1, dropout_end)
            coast_line.set_data(estimates[dropout_start:coast_end, 0],
                                estimates[dropout_start:coast_end, 1])
        else:
            coast_line.set_data([], [])

        if k >= dropout_end:
            post_line.set_data(estimates[dropout_end:k + 1, 0],
                               estimates[dropout_end:k + 1, 1])
        else:
            post_line.set_data([], [])

        if dropout_start <= k < dropout_end:
            phase_color, banner, banner_bg = "orange", "SENSOR DROPOUT — coasting on model", "navajowhite"
            ell.set_linestyle("--")
        elif k >= dropout_end:
            phase_color, banner, banner_bg = "green", "RECOVERY — measurements restored", "lightgreen"
            ell.set_linestyle("-")
        else:
            phase_color, banner, banner_bg = "blue", "TRACKING", "lightblue"
            ell.set_linestyle("-")

        est_dot.set_data([estimates[k, 0]], [estimates[k, 1]])
        est_dot.set_color(phase_color)

        # Confidence ellipse grows during dropout, shrinks on recovery
        w, h, angle = _ellipse_params(covariances[k, :2, :2])
        ell.set_center((estimates[k, 0], estimates[k, 1]))
        ell.width, ell.height, ell.angle = w, h, angle
        ell.set_edgecolor(phase_color)
        ell.set_visible(True)

        status_text.set_text(banner)
        status_text.get_bbox_patch().set_facecolor(banner_bg)

        sig_trail.set_data(t[s], pos_sigma[s])
        err_trail.set_data(t[s], err[s])
        time_marker.set_xdata([t[k], t[k]])

        return (meas_scat, true_trail, true_dot, pre_line, coast_line, post_line,
                est_dot, ell, status_text, sig_trail, err_trail, time_marker)

    _show_animation(fig, update, len(frame_indices))
