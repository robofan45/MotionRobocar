"""
Make the animated GIFs shown in the README.

Every demo runs with its plots switched on, drawn off screen at twice the
final size (so lines and text come out smooth when scaled down). A frame is
kept each time the demo calls plt.pause() (or for each frame of a matplotlib
FuncAnimation); demos that draw their result in one go are replayed in the
order they drew it, with long lines growing point by point. Frames are kept
evenly spaced, so the GIF plays at a steady speed, and the finished picture
is held for a moment before the GIF loops.

Each GIF gets a title strip and a progress bar. Demos that jump between
scenes, or grow a tree in big steps, get blended in-between frames
(crossfades); small 3D plots are zoomed in. If gifsicle is installed, it
shrinks the files without changing a pixel.

Each demo runs in its own Python process, so one demo cannot change another's
plots, and a demo that hangs is stopped after a timeout.

Usage:
    python utils/make_demo_gifs.py                  # every demo
    python utils/make_demo_gifs.py a_star dijkstra  # some demos
    python utils/make_demo_gifs.py --list

author: Mahim Trivedi
"""
import argparse
import contextlib
import dataclasses
import importlib
import io
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GIF_DIR = ROOT / "gifs"
WIDTH = 640            # GIF width in pixels
SUPERSAMPLE = 2        # plots are drawn at twice the size, then scaled down
MAX_UPSCALE = 1.25     # a small plot is never blown up more than this
MAX_FRAMES = 120       # evenly spaced animation steps kept per GIF
PLAYBACK_MS = 10000    # about how long one pass of the animation takes
MIN_FRAME_MS, MAX_FRAME_MS = 70, 700
CROSSFADE_MS = 50      # each blended in-between frame
FINAL_HOLD_MS = 3000   # the finished picture stays this long before looping
COLORS = 256           # GIF palette size
TITLE_PX, PROGRESS_PX = 40, 6  # strip heights at WIDTH (scaled with width)
STRIP_COLOR, TITLE_COLOR, SECTION_COLOR = (36, 41, 47), (255, 255, 255), \
    (154, 164, 178)
TRACK_COLOR, ACCENT_COLOR = (230, 232, 235), (47, 129, 247)
TIMEOUT_S = 900        # per demo


@dataclass(frozen=True)
class Demo:
    """How to run one README demo, and how its GIF should look."""
    module: str                            # import path
    title: str                             # shown in the title strip
    section: str = ""                      # filled in from SECTIONS
    entry: str | Callable = "main"         # function name, or f(module)
    flags: dict = field(default_factory=lambda: {"show_animation": True})
    max_frames: int = MAX_FRAMES
    width: int = WIDTH                     # wider for busy figures
    crossfade: int = 0                     # blended frames between steps
    zoom3d: float | None = None            # enlarge small 3D plots
    figsize: tuple | None = None           # for a plot drawn in one go


def _state_lattice(test_name):
    def run(module):
        # show the finished lattice, not every optimiser iteration
        sys.modules["ModelPredictiveTrajectoryGenerator."
                    "trajectory_generator"].show_animation = False
        getattr(module, test_name)()
    return run


def _bipedal_walk(module):
    # the same footsteps as the script's own __main__ block
    planner = module.BipedalPlanner()
    planner.set_ref_footsteps([[0.0, 0.2, 0.0], [0.3, 0.2, 0.0],
                               [0.3, 0.2, 0.2], [0.3, 0.2, 0.2],
                               [0.0, 0.2, 0.2]])
    planner.walk(plot=True)


NO_FLAGS: dict = {}
PP = "PathPlanning."
PT = "PathTracking."

# section -> {GIF name -> demo}, in README order
SECTIONS = {
    "Localization": {
        "extended_kalman_filter": Demo(
            "Localization.extended_kalman_filter.extended_kalman_filter",
            "Extended Kalman Filter localization"),
        "gps_imu_fusion": Demo(
            "Localization.gps_imu_fusion.gps_imu_fusion",
            "GPS/IMU fusion with bias estimation", width=960),
        "particle_filter": Demo(
            "Localization.particle_filter.particle_filter",
            "Particle filter localization"),
        "histogram_filter": Demo(
            "Localization.histogram_filter.histogram_filter",
            "Histogram filter localization"),
    },
    "Mapping": {
        "gaussian_grid_map": Demo(
            "Mapping.gaussian_grid_map.gaussian_grid_map",
            "Gaussian grid map", crossfade=10),
        "ray_casting_grid_map": Demo(
            "Mapping.ray_casting_grid_map.ray_casting_grid_map",
            "Ray casting grid map", crossfade=10),
        "lidar_to_grid_map": Demo(
            "Mapping.lidar_to_grid_map.lidar_to_grid_map",
            "Lidar to grid map", flags=NO_FLAGS, width=800,
            figsize=(7.2, 4.6)),
        "kmeans_clustering": Demo(
            "Mapping.kmeans_clustering.kmeans_clustering",
            "k-means object clustering"),
        "rectangle_fitting": Demo(
            "Mapping.rectangle_fitting.rectangle_fitting",
            "Rectangle fitting"),
    },
    "SLAM": {
        "icp_matching": Demo("SLAM.ICPMatching.icp_matching",
                             "Iterative Closest Point matching"),
        "fast_slam1": Demo("SLAM.FastSLAM1.fast_slam1", "FastSLAM 1.0"),
    },
    "Path planning": {
        "dynamic_window_approach": Demo(
            PP + "DynamicWindowApproach.dynamic_window_approach",
            "Dynamic Window Approach"),
        "dijkstra": Demo(PP + "Dijkstra.dijkstra", "Dijkstra search"),
        "a_star": Demo(PP + "AStar.a_star", "A* search"),
        "d_star": Demo(PP + "DStar.dstar", "D* replanning"),
        "d_star_lite": Demo(PP + "DStarLite.d_star_lite",
                            "D* Lite replanning"),
        "potential_field": Demo(
            PP + "PotentialFieldPlanning.potential_field_planning",
            "Potential field planning"),
        "coverage_path_planning": Demo(
            PP + "GridBasedSweepCPP.grid_based_sweep_coverage_path_planner",
            "Grid based coverage planning", flags={"do_animation": True}),
        "particle_swarm_optimization": Demo(
            PP + "ParticleSwarmOptimization.particle_swarm_optimization",
            "Particle swarm optimization"),
        "state_lattice_uniform": Demo(
            PP + "StateLatticePlanner.state_lattice_planner",
            "State lattice: uniform sampling",
            entry=_state_lattice("uniform_terminal_state_sampling_test1")),
        "state_lattice_biased": Demo(
            PP + "StateLatticePlanner.state_lattice_planner",
            "State lattice: biased polar sampling",
            entry=_state_lattice("biased_terminal_state_sampling_test1")),
        "state_lattice_lane": Demo(
            PP + "StateLatticePlanner.state_lattice_planner",
            "State lattice: lane sampling",
            entry=_state_lattice("lane_state_sampling_test1")),
        "probabilistic_road_map": Demo(
            PP + "ProbabilisticRoadMap.probabilistic_road_map",
            "Probabilistic roadmap (PRM)"),
        "rrt_star": Demo(PP + "RRTStar.rrt_star", "RRT*"),
        "rrt_star_reeds_shepp": Demo(
            PP + "RRTStarReedsShepp.rrt_star_reeds_shepp",
            "RRT* with Reeds-Shepp paths", crossfade=3),
        "lqr_rrt_star": Demo(PP + "LQRRRTStar.lqr_rrt_star", "LQR-RRT*",
                             crossfade=2),
        "quintic_polynomials": Demo(
            PP + "QuinticPolynomialsPlanner.quintic_polynomials_planner",
            "Quintic polynomials planning"),
        "reeds_shepp_path": Demo(
            PP + "ReedsSheppPath.reeds_shepp_path_planning",
            "Reeds-Shepp path"),
        "lqr_planner": Demo(PP + "LQRPlanner.lqr_planner",
                            "LQR path planning",
                            flags={"SHOW_ANIMATION": True}, crossfade=3),
        "frenet_optimal_trajectory": Demo(
            PP + "FrenetOptimalTrajectory.frenet_optimal_trajectory",
            "Optimal trajectory in a Frenet frame"),
    },
    "Path tracking": {
        "move_to_pose": Demo(PT + "move_to_pose.move_to_pose",
                             "Move to a pose"),
        "stanley_control": Demo(PT + "stanley_control.stanley_control",
                                "Stanley control"),
        "rear_wheel_feedback": Demo(
            PT + "rear_wheel_feedback_control.rear_wheel_feedback_control",
            "Rear wheel feedback control"),
        "lqr_speed_steer_control": Demo(
            PT + "lqr_speed_steer_control.lqr_speed_steer_control",
            "LQR speed and steering control"),
        "model_predictive_control": Demo(
            PT + "model_predictive_speed_and_steer_control."
            "model_predictive_speed_and_steer_control",
            "Model predictive control (MPC)"),
        "cgmres_nmpc": Demo(PT + "cgmres_nmpc.cgmres_nmpc",
                            "Nonlinear MPC with C-GMRES"),
    },
    "Arm navigation": {
        "n_joint_arm": Demo(
            "ArmNavigation.n_joint_arm_to_point_control."
            "n_joint_arm_to_point_control",
            "N joint arm to point control", entry="animation"),
        "arm_obstacle_navigation": Demo(
            "ArmNavigation.arm_obstacle_navigation.arm_obstacle_navigation",
            "Arm navigation around obstacles", flags=NO_FLAGS),
    },
    "Aerial navigation": {
        "drone_3d_trajectory": Demo(
            "AerialNavigation.drone_3d_trajectory_following."
            "drone_3d_trajectory_following",
            "Drone 3D trajectory following", zoom3d=1.3),
        "rocket_powered_landing": Demo(
            "AerialNavigation.rocket_powered_landing.rocket_powered_landing",
            "Rocket powered landing", zoom3d=1.6),
    },
    "Bipedal": {
        "bipedal_planner": Demo(
            "Bipedal.bipedal_planner.bipedal_planner",
            "Bipedal walking with an inverted pendulum",
            entry=_bipedal_walk, flags=NO_FLAGS, zoom3d=1.25),
    },
}
DEMOS = {name: dataclasses.replace(demo, section=section)
         for section, demos in SECTIONS.items()
         for name, demo in demos.items()}


def render(fig, pixels_wide, zoom3d=None):
    """Draw the figure, pixels_wide across, as an RGB image."""
    if not hasattr(fig.canvas, "buffer_rgba"):  # closed figures lose it
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    dpi = pixels_wide / fig.get_figwidth()
    if abs(fig.dpi - dpi) > 1e-3:
        fig.set_dpi(dpi)
    if zoom3d:
        for ax in fig.axes:
            if ax.name == "3d":
                ax.set_box_aspect(getattr(ax, "_box_aspect", None),
                                  zoom=zoom3d)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _is_open(fig):
    # figure numbers are reused after a close, so compare the figures
    from matplotlib._pylab_helpers import Gcf
    return any(m.canvas.figure is fig for m in Gcf.get_all_fig_managers())


class FrameRecorder:
    """Keeps between max_frames and 2 * max_frames evenly spaced frames of
    the figure that animates; only the kept steps are drawn. When that
    figure is closed and another one starts animating (one window per
    example), recording carries on in the new one."""

    def __init__(self, max_frames=MAX_FRAMES, width=WIDTH, zoom3d=None):
        self.max_frames = max_frames
        self.pixels = width * SUPERSAMPLE
        self.zoom3d = zoom3d
        self.frames = []      # (step number, PNG bytes)
        self.stride = 1
        self.calls = 0
        self.figure = None    # the animated figure
        self.shown = None     # the figure on screen at the last plt.show()

    def _snapshot(self, fig):
        return _png(render(fig, self.pixels, self.zoom3d))

    def offer(self, fig):
        """Called on every animation step of a figure."""
        if self.figure is None or (fig is not self.figure
                                   and not _is_open(self.figure)):
            self.figure = fig
        if fig is not self.figure:
            return
        if self.calls % self.stride == 0:
            self.frames.append((self.calls, self._snapshot(fig)))
            if len(self.frames) >= 2 * self.max_frames:
                self.frames = self.frames[::2]
                self.stride *= 2
        self.calls += 1

    def finish(self):
        """Add the finished picture if the last kept step is not it."""
        if self.figure is None:
            return
        # blitted animations mark their artists 'animated', which a normal
        # draw skips: draw them for the final picture
        animated = self.figure.findobj(lambda a: a.get_animated())
        for artist in animated:
            artist.set_animated(False)
        try:
            final = self._snapshot(self.figure)
        finally:
            for artist in animated:
                artist.set_animated(True)
        if not self.frames or self.frames[-1][1] != final:
            self.frames.append((self.calls, final))

    def images(self):
        """At most max_frames evenly spaced images, ending with the last."""
        data = [png for _, png in self.frames]
        if len(data) > self.max_frames:
            keep = np.linspace(0, len(data) - 1, self.max_frames)
            data = [data[i] for i in np.round(keep).astype(int)]
        return [Image.open(io.BytesIO(png)).convert("RGB") for png in data]


def _recording_writer(recorder):
    from matplotlib import animation

    class RecordingWriter(animation.AbstractMovieWriter):
        """Hands each FuncAnimation frame to the recorder."""

        def setup(self, fig, outfile, dpi=None):
            super().setup(fig, outfile, dpi)

        def grab_frame(self, **savefig_kwargs):
            recorder.offer(self.fig)

        def finish(self):
            pass

    return RecordingWriter(fps=10)


@contextlib.contextmanager
def _capturing(recorder):
    """Route plt.pause, plt.show and FuncAnimation to the recorder."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    animations = []
    original = (plt.pause, plt.show, animation.FuncAnimation.__init__)

    def pause(*args, **kwargs):
        recorder.offer(plt.gcf())

    def show(*args, **kwargs):
        while animations:
            animations.pop(0).save(os.devnull,
                                   writer=_recording_writer(recorder))
        recorder.shown = plt.gcf()

    def init(self, *args, **kwargs):
        original[2](self, *args, **kwargs)
        animations.append(self)

    plt.pause, plt.show = pause, show
    animation.FuncAnimation.__init__ = init
    try:
        yield
    finally:
        plt.pause, plt.show, animation.FuncAnimation.__init__ = original


def _is_long_line(artist):
    from matplotlib.lines import Line2D
    return (isinstance(artist, Line2D) and not hasattr(artist, "get_data_3d")
            and len(artist.get_xdata()) > 16)


def replay(fig, recorder):
    """Draw a finished figure again, artist by artist, in the order the demo
    drew it; long lines grow point by point."""
    axes = sorted(fig.axes, key=lambda ax: (round(ax.get_position().x0, 2),
                                            -ax.get_position().y1))
    for ax in axes:  # keep the finished axis limits while artists are hidden
        if ax.name != "3d":
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
    artists = [a for ax in axes for a in getattr(ax, "_children", [])
               if a.get_visible()]
    long_lines = sum(_is_long_line(a) for a in artists)
    grow = int(np.clip(recorder.max_frames // max(1, long_lines), 8, 60))
    for artist in artists:
        artist.set_visible(False)
    recorder.offer(fig)
    for artist in artists:
        artist.set_visible(True)
        if _is_long_line(artist):
            x, y = artist.get_xdata(), artist.get_ydata()
            for end in np.linspace(len(x) / grow, len(x), grow):
                artist.set_data(x[:int(end)], y[:int(end)])
                recorder.offer(fig)
            artist.set_data(x, y)
        recorder.offer(fig)
    recorder.finish()


def record(run, max_frames=MAX_FRAMES, width=WIDTH, zoom3d=None,
           figsize=None):
    """Run `run()` with its plotting captured; return the frames (drawn at
    SUPERSAMPLE times the GIF width). figsize resizes a plot that was drawn
    in one go before it is replayed."""
    import matplotlib.pyplot as plt

    recorder = FrameRecorder(max_frames, width, zoom3d)
    try:
        with _capturing(recorder):
            run()
        if recorder.frames:
            recorder.finish()
        else:  # drew everything at once: replay it
            fig = recorder.shown or plt.gcf()
            if figsize:
                fig.set_size_inches(figsize)
            replay(fig, recorder)
        return recorder.images()
    finally:
        plt.close("all")


def _trim(frames, pad=12):
    """Crop the blank margin that no frame draws into."""
    background = Image.new("RGB", frames[0].size, frames[0].getpixel((0, 0)))
    boxes = [ImageChops.difference(f, background).getbbox() for f in frames]
    boxes = [b for b in boxes if b]
    if not boxes:
        return frames
    width, height = frames[0].size
    box = (max(0, min(b[0] for b in boxes) - pad),
           max(0, min(b[1] for b in boxes) - pad),
           min(width, max(b[2] for b in boxes) + pad),
           min(height, max(b[3] for b in boxes) + pad))
    return [f.crop(box) for f in frames]


def _fit(frames, width):
    """Scale to the GIF width (tall plots to a 4:4.5 box), never blowing a
    small plot up more than MAX_UPSCALE."""
    w, h = frames[0].size
    scale = min(width / w, 1.125 * width / h, MAX_UPSCALE)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if size == (w, h):
        return frames
    return [f.resize(size, Image.Resampling.LANCZOS) for f in frames]


@cache
def _font(bold, size):
    from matplotlib import get_data_path
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(get_data_path(), "fonts", "ttf", name)),
                              size)


def _decorate(frame, title, section, progress, progress_bar):
    """Add the title strip on top and the progress bar underneath."""
    w, h = frame.size
    k = w / WIDTH
    top = round(TITLE_PX * k) if title else 0
    bottom = max(2, round(PROGRESS_PX * k)) if progress_bar else 0
    canvas = Image.new("RGB", (w, top + h + bottom), "white")
    canvas.paste(frame, (0, top))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.rectangle((0, 0, w, top - 1), fill=STRIP_COLOR)
        pad = round(14 * k)
        size = max(8, round(19 * k))
        while size > 8 and draw.textlength(title, _font(True, size)) \
                > w - 2 * pad:
            size -= 1
        draw.text((pad, top / 2), title, font=_font(True, size),
                  fill=TITLE_COLOR, anchor="lm")
        small = _font(False, max(8, round(14 * k)))
        room = w - 3 * pad - draw.textlength(title, _font(True, size))
        if section and draw.textlength(section, small) <= room:
            draw.text((w - pad, top / 2), section, font=small,
                      fill=SECTION_COLOR, anchor="rm")
    if progress_bar:
        y = top + h
        draw.rectangle((0, y, w, y + bottom), fill=TRACK_COLOR)
        if progress > 0:
            draw.rectangle((0, y, round(progress * w) - 1, y + bottom),
                           fill=ACCENT_COLOR)
    return canvas


def write_gif(frames, path, *, width=WIDTH, title=None, section=None,
              crossfade=0, progress_bar=True):
    """Save frames as a looping GIF: blank margins trimmed, scaled to the
    GIF width, optional crossfades, title strip and progress bar, one shared
    colour palette, and the last frame held."""
    size = frames[0].size
    frames = [f if f.size == size else f.resize(size) for f in frames]
    frames = _fit(_trim(frames), width)
    last = max(1, len(frames) - 1)
    step = int(min(MAX_FRAME_MS, max(MIN_FRAME_MS, PLAYBACK_MS / last)))
    timeline = []  # (image, duration in ms, progress 0..1)
    for i, frame in enumerate(frames):
        if crossfade and i > 0:
            for j in range(1, crossfade + 1):
                share = j / (crossfade + 1)
                timeline.append((Image.blend(frames[i - 1], frame, share),
                                 CROSSFADE_MS, (i - 1 + share) / last))
        timeline.append((frame, step, i / last))
    timeline[-1] = (timeline[-1][0], FINAL_HOLD_MS, 1.0)
    images = [_decorate(image, title, section, progress, progress_bar)
              for image, _, progress in timeline]
    sample = images[::max(1, len(images) // 12)] + [images[-1]]
    w, h = images[0].size
    strip = Image.new("RGB", (w, h * len(sample)))
    for i, image in enumerate(sample):
        strip.paste(image, (0, i * h))
    palette = strip.quantize(COLORS, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
    paletted = [image.quantize(palette=palette, dither=Image.Dither.NONE)
                for image in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(path, save_all=True, append_images=paletted[1:],
                     duration=[d for _, d, _ in timeline], loop=0, disposal=1)
    if shutil.which("gifsicle"):  # lossless: same pixels, smaller file
        subprocess.run(["gifsicle", "--batch", "-O3", str(path)], check=True)


def make_gif(name):
    """Run one demo (in this process) and write gifs/<name>.gif."""
    import matplotlib
    matplotlib.use("Agg")
    demo = DEMOS[name]
    sys.path.insert(0, str(ROOT))
    with contextlib.redirect_stdout(io.StringIO()):  # demos print a lot
        module = importlib.import_module(demo.module)
        os.chdir(Path(module.__file__).parent)  # demos read files next to them
        for attr, value in demo.flags.items():
            setattr(module, attr, value)
        random.seed(1)  # demos seed themselves; this covers helper randomness
        np.random.seed(1)

        def run():
            if isinstance(demo.entry, str):
                getattr(module, demo.entry)()
            else:
                demo.entry(module)

        frames = record(run, demo.max_frames, demo.width, demo.zoom3d,
                        demo.figsize)
    path = GIF_DIR / f"{name}.gif"
    write_gif(frames, path, width=demo.width, title=demo.title,
              section=demo.section, crossfade=demo.crossfade)
    with Image.open(path) as gif:
        return gif.n_frames, gif.size, path.stat().st_size


def _run_in_child(name):
    start = time.perf_counter()
    try:
        res = subprocess.run([sys.executable, __file__, "--child", name],
                             capture_output=True, text=True, timeout=TIMEOUT_S,
                             env={**os.environ, "MPLBACKEND": "Agg"})
    except subprocess.TimeoutExpired:
        return name, False, f"timed out after {TIMEOUT_S} s"
    seconds = time.perf_counter() - start
    if res.returncode != 0:
        return name, False, res.stderr.strip().splitlines()[-1:]
    return name, True, f"{res.stdout.strip()} in {seconds:.0f} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="demos to make (default: all)")
    parser.add_argument("--list", action="store_true", help="list the demos")
    parser.add_argument("--jobs", type=int, default=2,
                        help="demos to run at the same time")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        frames, size, nbytes = make_gif(args.child)
        print(f"{frames} frames, {size[0]}x{size[1]}, {nbytes / 1e6:.2f} MB")
        return 0
    if args.list:
        print("\n".join(DEMOS))
        return 0
    names = args.names or list(DEMOS)
    unknown = sorted(set(names) - set(DEMOS))
    if unknown:
        parser.error(f"unknown demos: {', '.join(unknown)}")
    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for name, ok, detail in pool.map(_run_in_child, names):
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
            failed += not ok
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
