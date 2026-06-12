"""Plotting and video-rendering helpers for N3 — LiDAR 3D Detection & Tracking.

All visualization code lives in this module so the notebook cells stay
focused on the perception math: ground removal, clustering, and tracking.
Nothing in here changes the algorithms — every function just draws.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio

track_colors = plt.cm.tab20(np.linspace(0, 1, 20))

# Prediction arrows get one fixed high-contrast color (instead of the track
# color) so they're easy to spot against the boxes and trails
PRED_COLOR = 'magenta'

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def get_box_corners(bmin, bmax):
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def show_raw_bev(img, pts_roi):
    """Camera image next to the bird's-eye view of the ROI-cropped point cloud."""
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 5))

    ax_cam.imshow(img)
    ax_cam.set_title("Camera 2 — Frame 0", fontsize=13)
    ax_cam.axis('off')

    ax_bev.scatter(-pts_roi[:, 1], pts_roi[:, 0], s=0.1, c=pts_roi[:, 2],
                   cmap='viridis', alpha=0.5, vmin=-2, vmax=1)
    ax_bev.set_xlim(-25, 25)
    ax_bev.set_ylim(0, 60)
    ax_bev.set_xlabel(r'$\leftarrow$ Left          Right $\rightarrow$', fontsize=11)
    ax_bev.set_ylabel(r'Forward (m) $\rightarrow$', fontsize=11)
    ax_bev.set_title(f"Bird's-Eye View — {len(pts_roi):,} points in ROI", fontsize=13)
    ax_bev.set_aspect('equal')
    ax_bev.plot(0, 0, 'r^', markersize=12, label='Ego vehicle')
    ax_bev.legend(fontsize=10)
    ax_bev.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def show_normals(vis_pts, normals, verticality):
    """Surface normals as arrows, colored by verticality |n_z| (green=ground-like)."""
    fig = plt.figure(figsize=(16, 6))

    # Left: 3D perspective with normal arrows
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                 c=verticality, cmap='RdYlGn', s=8, alpha=0.6, vmin=0, vmax=1)
    arrow_len = 0.4
    ax3d.quiver(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                normals[:, 0] * arrow_len,
                normals[:, 1] * arrow_len,
                normals[:, 2] * arrow_len,
                color=plt.cm.RdYlGn(verticality), alpha=0.5, linewidth=0.7,
                arrow_length_ratio=0.3)
    ax3d.set_xlabel('X — Forward (m)')
    ax3d.set_ylabel('Y — Left (m)')
    ax3d.set_zlabel('Z — Up (m)')
    ax3d.set_title('Surface Normals — 3D View', fontsize=13)
    ax3d.view_init(elev=25, azim=-60)

    # Right: side view (X vs Z) — clearest way to see ground vs obstacle normals
    ax_side = fig.add_subplot(122)
    sc2 = ax_side.scatter(vis_pts[:, 0], vis_pts[:, 2], c=verticality,
                          cmap='RdYlGn', s=12, alpha=0.5, vmin=0, vmax=1)
    ax_side.quiver(vis_pts[:, 0], vis_pts[:, 2],
                   normals[:, 0], normals[:, 2],
                   verticality, cmap='RdYlGn', clim=(0, 1),
                   scale=25, width=0.003, alpha=0.7)
    ax_side.set_xlabel('X — Forward (m)', fontsize=11)
    ax_side.set_ylabel('Z — Height (m)', fontsize=11)
    ax_side.set_title('Surface Normals — Side View', fontsize=13)
    ax_side.set_aspect('equal')
    ax_side.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc2, ax=ax_side, shrink=0.8, pad=0.02)
    cbar.set_label('Verticality  |n_z|', fontsize=10)
    cbar.ax.set_yticks([0, 0.5, 0.85, 1.0])
    cbar.ax.set_yticklabels(['0 (horizontal)', '0.5', '0.85 (threshold)', '1.0 (vertical)'])

    plt.tight_layout()
    plt.show()


def show_ground_removal(img, ground, non_ground, uv_g, vis_g, uv_ng, vis_ng):
    """Ground vs non-ground points, projected onto the camera image and in BEV."""
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 5),
                                         gridspec_kw={'width_ratios': [1.3, 1]})

    ax_cam.imshow(img)
    ax_cam.scatter(uv_g[vis_g, 0], uv_g[vis_g, 1], s=0.3, c='cyan', alpha=0.25,
                   label=f'Ground ({vis_g.sum():,})')
    ax_cam.scatter(uv_ng[vis_ng, 0], uv_ng[vis_ng, 1], s=0.8, c='red', alpha=0.4,
                   label=f'Non-ground ({vis_ng.sum():,})')
    ax_cam.legend(fontsize=10, loc='upper right', markerscale=10)
    ax_cam.set_title('Ground Removal — Camera Projection', fontsize=13)
    ax_cam.axis('off')

    ax_bev.scatter(-ground[:, 1], ground[:, 0], s=0.3, c='gray', alpha=0.3,
                   label=f'Ground ({len(ground):,})')
    ax_bev.scatter(-non_ground[:, 1], non_ground[:, 0], s=0.5, c='red', alpha=0.3,
                   label=f'Non-ground ({len(non_ground):,})')
    ax_bev.set_xlim(-25, 25)
    ax_bev.set_ylim(0, 60)
    ax_bev.set_aspect('equal')
    ax_bev.legend(fontsize=10, markerscale=10)
    ax_bev.set_title('Ground Removal — BEV', fontsize=13)
    ax_bev.set_xlabel(r'$\leftarrow$ Left     Right $\rightarrow$')
    ax_bev.set_ylabel('Forward (m)')
    ax_bev.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def show_detections_bev(non_ground, detections, title=""):
    """BEV scatter of non-ground points with detected boxes, dims, and point counts."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(-non_ground[:, 1], non_ground[:, 0], s=0.3, c='lightgray', alpha=0.3)

    for i, det in enumerate(detections):
        c = track_colors[i % 20]
        bmin, bmax = det['bbox_min'], det['bbox_max']
        rect = plt.Rectangle((-bmax[1], bmin[0]),
                             bmax[1] - bmin[1], bmax[0] - bmin[0],
                             linewidth=2, edgecolor=c, facecolor=c, alpha=0.15)
        ax.add_patch(rect)
        cx, cy = -det['centroid'][1], det['centroid'][0]
        ax.text(cx, cy + 0.8,
                f"{det['dims'][0]:.1f} x {det['dims'][1]:.1f} x {det['dims'][2]:.1f} m\n"
                f"{det['n_points']} pts",
                fontsize=7, ha='center', color=c, fontweight='bold')

    ax.plot(0, 0, 'r^', markersize=12)
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 60)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\leftarrow$ Left     Right $\rightarrow$')
    ax.set_ylabel('Forward (m)')
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def show_road_filtering(img, road_mask, all_dets, road_dets):
    """Road segmentation overlay next to a BEV of kept vs filtered detections."""
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(16, 6),
                                         gridspec_kw={'width_ratios': [1.2, 1]})
    overlay = img.copy()
    overlay[road_mask] = (0.5 * overlay[road_mask] +
                          0.5 * np.array([0, 180, 0])).astype(np.uint8)
    ax_cam.imshow(overlay)
    ax_cam.set_title('Road Segmentation (SegFormer from N1)', fontsize=13)
    ax_cam.axis('off')

    for det in all_dets:
        bmin, bmax = det['bbox_min'], det['bbox_max']
        rect = plt.Rectangle((-bmax[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             linewidth=1.5, edgecolor='gray', facecolor='none',
                             linestyle='--')
        ax_bev.add_patch(rect)

    for i, det in enumerate(road_dets):
        c = track_colors[i % 20]
        bmin, bmax = det['bbox_min'], det['bbox_max']
        rect = plt.Rectangle((-bmax[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             linewidth=2, edgecolor=c, facecolor=c, alpha=0.15)
        ax_bev.add_patch(rect)
        ax_bev.text(-det['centroid'][1], det['centroid'][0] + 1,
                    f"{det['dims'][0]:.1f}×{det['dims'][1]:.1f}m",
                    color=c, fontsize=8, ha='center', fontweight='bold')

    ax_bev.plot(0, 0, 'r^', markersize=12)
    ax_bev.set_xlim(-25, 25)
    ax_bev.set_ylim(0, 60)
    ax_bev.set_aspect('equal')
    ax_bev.set_title(f'Detections: {len(all_dets)} total → {len(road_dets)} on road',
                     fontsize=13)
    ax_bev.set_xlabel(r'$\leftarrow$ Left     Right $\rightarrow$')
    ax_bev.set_ylabel('Forward (m)')
    ax_bev.grid(True, alpha=0.2)
    ax_bev.legend(handles=[
        mpatches.Patch(edgecolor='gray', facecolor='none', linestyle='--',
                       label='Off-road (filtered)'),
        mpatches.Patch(edgecolor='tab:blue', facecolor='tab:blue', alpha=0.3,
                       label='On-road (kept)'),
    ], fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_bev_frame(ax, fr, title="", road_pts=None, road_clr=None):
    """One world-frame BEV panel, centered on the ego vehicle: road surface,
    ego triangle, track boxes, trajectory history, and prediction arrows."""
    ego = fr.get('ego_pos', np.zeros(3))
    heading = fr.get('ego_heading', 0.0)

    cx, cy = -ego[1], ego[0]
    ax.set_xlim(cx - 20, cx + 20)
    ax.set_ylim(cy - 10, cy + 45)
    ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    ax.grid(True, alpha=0.15)

    # Road surface (transform ego-frame points to world frame)
    if road_pts is not None and len(road_pts) > 0:
        T_ew = fr.get('T_ego_to_world')
        if T_ew is not None:
            rp_h = np.hstack([road_pts, np.ones((len(road_pts), 1))])
            rp_w = (T_ew @ rp_h.T).T[:, :3]
        else:
            rp_w = road_pts
        c = road_clr if road_clr is not None else '#a5d6a7'
        ax.scatter(-rp_w[:, 1], rp_w[:, 0], s=1.5, c=c,
                   alpha=0.6, zorder=1, edgecolors='none')

    # Ego vehicle — rotated triangle showing heading
    s = 1.5
    tri = np.array([[s, 0], [-s, -s * 0.7], [-s, s * 0.7]])
    R = np.array([[np.cos(heading), -np.sin(heading)],
                  [np.sin(heading),  np.cos(heading)]])
    tri_r = (R @ tri.T).T
    tri_bev = np.column_stack([-tri_r[:, 1] + cx, tri_r[:, 0] + cy])
    ego_patch = mpatches.Polygon(tri_bev, closed=True, facecolor='red',
                                 edgecolor='darkred', linewidth=1.5, zorder=10)
    ax.add_patch(ego_patch)
    ax.set_title(title, fontsize=11)

    for tr in fr['tracks']:
        c = track_colors[tr['id'] % 20]
        dims = tr['bbox_max'] - tr['bbox_min']
        half = dims / 2
        bmin_w = tr['pos'] - half
        bmax_w = tr['pos'] + half
        rect = plt.Rectangle((-bmax_w[1], bmin_w[0]),
                             bmax_w[1] - bmin_w[1], bmax_w[0] - bmin_w[0],
                             linewidth=2, edgecolor=c, facecolor=c, alpha=0.25,
                             zorder=5)
        ax.add_patch(rect)

        hist = np.array(tr['history'])
        if len(hist) > 1:
            ax.plot(-hist[:, 1], hist[:, 0], color=c, linewidth=2.5,
                    alpha=0.8, zorder=6)

        if tr['predictions']:
            last_p = tr['predictions'][-1]
            ax.annotate('', xy=(-last_p[1], last_p[0]),
                        xytext=(-tr['pos'][1], tr['pos'][0]),
                        arrowprops=dict(arrowstyle='->', color=PRED_COLOR,
                                        lw=2.5, mutation_scale=15),
                        zorder=7)

        ax.text(-tr['pos'][1], tr['pos'][0] + 1.2, f"{tr['id']}",
                color=c, fontsize=7, ha='center', fontweight='bold',
                zorder=8)


def show_tracking_frames(frame_results, road_bev, road_bev_colors, ego_positions,
                         sample_frames, num_frames):
    """2x2 grid of BEV snapshots at the given sample frames."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, fi in zip(axes.flat, sample_frames):
        fi = min(fi, num_frames - 1)
        plot_bev_frame(ax, frame_results[fi],
                       f"Frame {fi} — {len(frame_results[fi]['tracks'])} tracks",
                       road_pts=road_bev[fi], road_clr=road_bev_colors[fi])
        trail = ego_positions[:fi + 1]
        ax.plot(-trail[:, 1], trail[:, 0], 'r-', linewidth=1.5, alpha=0.3, zorder=9)
        ax.set_xlabel(r'$\leftarrow$ Left     Right $\rightarrow$')
        ax.set_ylabel('Forward (m)')
    plt.tight_layout()
    plt.show()


def render_tracking_video(data, frame_results, road_bev, road_bev_colors,
                          ego_positions, M_proj, num_frames,
                          out_path='videos/lidar_tracking.mp4', fps=10):
    """Side-by-side video: camera with projected 3D boxes + prediction arrows
    on the left, ego-centered BEV with trails and arrows on the right."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, quality=7)

    print(f"Rendering {num_frames} frames to video...")
    for fi in range(num_frames):
        if fi % 20 == 0:
            print(f"  Rendering frame {fi}/{num_frames}")

        fig, (ax_cam, ax_bev) = plt.subplots(
            1, 2, figsize=(16, 5.5), dpi=100,
            gridspec_kw={'width_ratios': [1.2, 1]})

        # --- Camera panel ---
        img = np.array(data.get_cam2(fi))
        ax_cam.imshow(img)
        ax_cam.set_title(f'Frame {fi} / {num_frames - 1}', fontsize=11)
        ax_cam.axis('off')

        fr = frame_results[fi]
        h_img, w_img = img.shape[:2]
        T_w2e = fr['T_world_to_ego']

        for tr in fr['tracks']:
            c = track_colors[tr['id'] % 20]
            corners = get_box_corners(tr['bbox_min'], tr['bbox_max'])
            hom = np.hstack([corners, np.ones((8, 1))])
            proj = (M_proj @ hom.T).T
            depth = proj[:, 2]
            valid = depth > 0.5
            pts2d = np.zeros((8, 2))
            if valid.any():
                pts2d[valid] = proj[valid, :2] / proj[valid, 2:3]

            in_img = (valid &
                      (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w_img) &
                      (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h_img))

            for ei, ej in BOX_EDGES:
                if in_img[ei] and in_img[ej]:
                    ax_cam.plot([pts2d[ei, 0], pts2d[ej, 0]],
                                [pts2d[ei, 1], pts2d[ej, 1]],
                                color=c, linewidth=1.5)

            if in_img.any():
                cx2d = pts2d[in_img, 0].mean()
                cy2d = pts2d[in_img, 1].min() - 6
                ax_cam.text(cx2d, max(cy2d, 5), f'ID {tr["id"]}',
                            color=c, fontsize=8, fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round,pad=0.15',
                                      facecolor='black', alpha=0.5))

            # Prediction arrow: world frame → ego frame → camera projection
            if tr['predictions']:
                pred_5 = tr['predictions'][-1]
                cur_ego = (T_w2e @ np.array([*tr['pos'], 1.0]))[:3]
                pred_ego = (T_w2e @ np.array([*pred_5, 1.0]))[:3]
                cp = M_proj @ np.array([*cur_ego, 1.0])
                pp = M_proj @ np.array([*pred_ego, 1.0])
                if cp[2] > 0.5 and pp[2] > 0.5:
                    cu, cv = cp[0] / cp[2], cp[1] / cp[2]
                    pu, pv = pp[0] / pp[2], pp[1] / pp[2]
                    if (0 <= cu < w_img and 0 <= cv < h_img and
                            0 <= pu < w_img and 0 <= pv < h_img):
                        ax_cam.annotate('', xy=(pu, pv), xytext=(cu, cv),
                                        arrowprops=dict(arrowstyle='->',
                                                        color=PRED_COLOR,
                                                        lw=2.5, mutation_scale=15))

        # --- BEV panel (world frame, centered on ego) ---
        plot_bev_frame(ax_bev, fr, f'{len(fr["tracks"])} tracked objects',
                       road_pts=road_bev[fi], road_clr=road_bev_colors[fi])
        trail = ego_positions[:fi + 1]
        ax_bev.plot(-trail[:, 1], trail[:, 0], 'r-', linewidth=1.5, alpha=0.3, zorder=9)
        ax_bev.set_xlabel(r'$\leftarrow$ Left     Right $\rightarrow$', fontsize=9)
        ax_bev.set_ylabel('Forward (m)', fontsize=9)

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame_img = np.frombuffer(fig.canvas.buffer_rgba(),
                                  dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        writer.append_data(frame_img)
        plt.close(fig)

    writer.close()
    print(f"\nSaved: {out_path}  ({num_frames} frames @ {fps} fps)")
