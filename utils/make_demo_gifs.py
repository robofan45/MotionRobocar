"""
Make the animated GIFs shown in the README.

Every demo runs with its plots switched on, drawn off screen. A frame is kept
each time the demo calls plt.pause() (or for each frame of a matplotlib
FuncAnimation). Demos that draw their result in one go are replayed in the
order they drew it, with long lines growing point by point. Frames are kept
evenly spaced, and the finished picture is held for a moment before the GIF
loops.

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
import importlib
import io
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
GIF_DIR = ROOT / "gifs"
WIDTH = 480            # GIF width in pixels
MAX_FRAMES = 60        # evenly spaced frames kept per GIF
LOOP_MS = 7000         # about how long one pass of the animation takes
MIN_FRAME_MS, MAX_FRAME_MS = 60, 1000
FINAL_HOLD_MS = 2500   # the finished picture stays this long before looping
COLORS = 128           # GIF palette size
GROW_STEPS = 12        # frames used to draw one long line in a replay
TIMEOUT_S = 900        # per demo


@dataclass(frozen=True)
class Demo:
    """How to run one README demo."""
    module: str                                   # import path
    entry: str | Callable = "main"                # function name, or f(module)
    flags: dict = field(default_factory=lambda: {"show_animation": True})
    max_frames: int = MAX_FRAMES
    width: int = WIDTH                            # wider for busy figures


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

# GIF name -> demo, in README order
DEMOS = {
    # Localization
    "extended_kalman_filter": Demo(
        "Localization.extended_kalman_filter.extended_kalman_filter"),
    "gps_imu_fusion": Demo("Localization.gps_imu_fusion.gps_imu_fusion",
                           width=760),
    "particle_filter": Demo("Localization.particle_filter.particle_filter"),
    "histogram_filter": Demo("Localization.histogram_filter.histogram_filter"),
    # Mapping
    "gaussian_grid_map": Demo("Mapping.gaussian_grid_map.gaussian_grid_map"),
    "ray_casting_grid_map": Demo(
        "Mapping.ray_casting_grid_map.ray_casting_grid_map"),
    "lidar_to_grid_map": Demo("Mapping.lidar_to_grid_map.lidar_to_grid_map",
                              flags=NO_FLAGS, width=720),
    "kmeans_clustering": Demo("Mapping.kmeans_clustering.kmeans_clustering"),
    "rectangle_fitting": Demo("Mapping.rectangle_fitting.rectangle_fitting"),
    # SLAM
    "icp_matching": Demo("SLAM.ICPMatching.icp_matching"),
    "fast_slam1": Demo("SLAM.FastSLAM1.fast_slam1"),
    # Path planning
    "dynamic_window_approach": Demo(
        "PathPlanning.DynamicWindowApproach.dynamic_window_approach"),
    "dijkstra": Demo("PathPlanning.Dijkstra.dijkstra"),
    "a_star": Demo("PathPlanning.AStar.a_star"),
    "d_star": Demo("PathPlanning.DStar.dstar"),
    "d_star_lite": Demo("PathPlanning.DStarLite.d_star_lite"),
    "potential_field": Demo(
        "PathPlanning.PotentialFieldPlanning.potential_field_planning"),
    "coverage_path_planning": Demo(
        "PathPlanning.GridBasedSweepCPP.grid_based_sweep_coverage_path_planner",
        flags={"do_animation": True}),
    "particle_swarm_optimization": Demo(
        "PathPlanning.ParticleSwarmOptimization.particle_swarm_optimization"),
    "state_lattice_uniform": Demo(
        "PathPlanning.StateLatticePlanner.state_lattice_planner",
        entry=_state_lattice("uniform_terminal_state_sampling_test1")),
    "state_lattice_biased": Demo(
        "PathPlanning.StateLatticePlanner.state_lattice_planner",
        entry=_state_lattice("biased_terminal_state_sampling_test1")),
    "state_lattice_lane": Demo(
        "PathPlanning.StateLatticePlanner.state_lattice_planner",
        entry=_state_lattice("lane_state_sampling_test1")),
    "probabilistic_road_map": Demo(
        "PathPlanning.ProbabilisticRoadMap.probabilistic_road_map"),
    "rrt_star": Demo("PathPlanning.RRTStar.rrt_star"),
    "rrt_star_reeds_shepp": Demo(
        "PathPlanning.RRTStarReedsShepp.rrt_star_reeds_shepp"),
    "lqr_rrt_star": Demo("PathPlanning.LQRRRTStar.lqr_rrt_star"),
    "quintic_polynomials": Demo(
        "PathPlanning.QuinticPolynomialsPlanner.quintic_polynomials_planner"),
    "reeds_shepp_path": Demo(
        "PathPlanning.ReedsSheppPath.reeds_shepp_path_planning"),
    "lqr_planner": Demo("PathPlanning.LQRPlanner.lqr_planner",
                        flags={"SHOW_ANIMATION": True}),
    "frenet_optimal_trajectory": Demo(
        "PathPlanning.FrenetOptimalTrajectory.frenet_optimal_trajectory"),
    # Path tracking
    "move_to_pose": Demo("PathTracking.move_to_pose.move_to_pose"),
    "stanley_control": Demo("PathTracking.stanley_control.stanley_control"),
    "rear_wheel_feedback": Demo(
        "PathTracking.rear_wheel_feedback_control.rear_wheel_feedback_control"),
    "lqr_speed_steer_control": Demo(
        "PathTracking.lqr_speed_steer_control.lqr_speed_steer_control"),
    "model_predictive_control": Demo(
        "PathTracking.model_predictive_speed_and_steer_control."
        "model_predictive_speed_and_steer_control"),
    "cgmres_nmpc": Demo("PathTracking.cgmres_nmpc.cgmres_nmpc"),
    # Arm navigation
    "n_joint_arm": Demo(
        "ArmNavigation.n_joint_arm_to_point_control."
        "n_joint_arm_to_point_control", entry="animation"),
    "arm_obstacle_navigation": Demo(
        "ArmNavigation.arm_obstacle_navigation.arm_obstacle_navigation",
        flags=NO_FLAGS),
    # Aerial navigation
    "drone_3d_trajectory": Demo(
        "AerialNavigation.drone_3d_trajectory_following."
        "drone_3d_trajectory_following"),
    "rocket_powered_landing": Demo(
        "AerialNavigation.rocket_powered_landing.rocket_powered_landing"),
    # Bipedal
    "bipedal_planner": Demo("Bipedal.bipedal_planner.bipedal_planner",
                            entry=_bipedal_walk, flags=NO_FLAGS),
}


def render(fig, width=WIDTH):
    """The figure as an RGB image, scaled to the given width."""
    if not hasattr(fig.canvas, "buffer_rgba"):  # closed figures lose it
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    if image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _same(a, b):
    return a.size == b.size and ImageChops.difference(a, b).getbbox() is None


class FrameRecorder:
    """Keeps between max_frames and 2 * max_frames evenly spaced frames of
    the first figure that animates; only the kept steps are drawn."""

    def __init__(self, max_frames=MAX_FRAMES, width=WIDTH):
        self.max_frames = max_frames
        self.width = width
        self.frames = []      # (step number, image)
        self.stride = 1
        self.calls = 0
        self.figure = None    # the animated figure
        self.shown = None     # the figure on screen at the last plt.show()

    def offer(self, fig):
        """Called on every animation step of a figure."""
        if self.figure is None:
            self.figure = fig
        if fig is not self.figure:
            return
        if self.calls % self.stride == 0:
            self.frames.append((self.calls, render(fig, self.width)))
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
            final = render(self.figure, self.width)
        finally:
            for artist in animated:
                artist.set_animated(True)
        if not self.frames or not _same(self.frames[-1][1], final):
            self.frames.append((self.calls, final))

    def images(self):
        """At most max_frames evenly spaced images, ending with the last."""
        images = [image for _, image in self.frames]
        if len(images) > self.max_frames:
            keep = np.linspace(0, len(images) - 1, self.max_frames)
            images = [images[i] for i in np.round(keep).astype(int)]
        return images


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


def replay(fig, recorder):
    """Draw a finished figure again, artist by artist, in the order the demo
    drew it; long lines grow point by point."""
    from matplotlib.lines import Line2D

    axes = sorted(fig.axes, key=lambda ax: (round(ax.get_position().x0, 2),
                                            -ax.get_position().y1))
    for ax in axes:  # keep the finished axis limits while artists are hidden
        if ax.name != "3d":
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
    artists = [a for ax in axes for a in getattr(ax, "_children", [])
               if a.get_visible()]
    for artist in artists:
        artist.set_visible(False)
    recorder.offer(fig)
    for artist in artists:
        artist.set_visible(True)
        if (isinstance(artist, Line2D) and not hasattr(artist, "get_data_3d")
                and len(artist.get_xdata()) > 2 * GROW_STEPS):
            x, y = artist.get_xdata(), artist.get_ydata()
            for end in np.linspace(len(x) / GROW_STEPS, len(x), GROW_STEPS):
                artist.set_data(x[:int(end)], y[:int(end)])
                recorder.offer(fig)
            artist.set_data(x, y)
        recorder.offer(fig)
    recorder.finish()


def record(run, max_frames=MAX_FRAMES, width=WIDTH):
    """Run `run()` with its plotting captured; return the GIF frames."""
    import matplotlib.pyplot as plt

    recorder = FrameRecorder(max_frames, width)
    try:
        with _capturing(recorder):
            run()
        if recorder.frames:
            recorder.finish()
        else:  # drew everything at once: replay it
            replay(recorder.shown or plt.gcf(), recorder)
        return recorder.images()
    finally:
        plt.close("all")


def _trim(frames, pad=6):
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


def write_gif(frames, path):
    """Save frames as a looping GIF with one shared colour palette."""
    size = frames[0].size
    frames = _trim([f if f.size == size else f.resize(size) for f in frames])
    size = frames[0].size
    sample = frames[::max(1, len(frames) // 8)] + [frames[-1]]
    strip = Image.new("RGB", (size[0], size[1] * len(sample)))
    for i, frame in enumerate(sample):
        strip.paste(frame, (0, i * size[1]))
    palette = strip.quantize(COLORS, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE)
                for f in frames]
    step = LOOP_MS / max(1, len(frames) - 1)
    step = int(min(MAX_FRAME_MS, max(MIN_FRAME_MS, step)))
    durations = [step] * (len(frames) - 1) + [FINAL_HOLD_MS]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(path, save_all=True, append_images=paletted[1:],
                     duration=durations, loop=0, disposal=1)


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

        frames = record(run, demo.max_frames, demo.width)
    path = GIF_DIR / f"{name}.gif"
    write_gif(frames, path)
    return len(frames), path.stat().st_size


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
        frames, size = make_gif(args.child)
        print(f"{frames} frames, {size / 1e6:.2f} MB")
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
