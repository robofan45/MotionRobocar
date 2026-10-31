"""
Unit tests for utils/make_demo_gifs.py, the tool that records the README GIFs.

author: Mahim Trivedi
"""
import ast
import importlib.util
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from PIL import Image

import conftest  # noqa: F401  (adds the project root to sys.path)
from utils import make_demo_gifs as gifs

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _small_figure():
    fig = plt.figure(figsize=(2, 1.5), dpi=40)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    return fig


def test_recorder_keeps_evenly_spaced_frames():
    fig = _small_figure()
    draws = []
    original_draw = fig.canvas.draw
    fig.canvas.draw = lambda *a, **k: (draws.append(1), original_draw())
    rec = gifs.FrameRecorder(max_frames=10, width=60)
    for _ in range(1000):
        rec.offer(fig)
    plt.close(fig)
    calls = [call for call, _ in rec.frames]
    steps = set(np.diff(calls))
    assert 10 <= len(calls) < 20
    assert calls[0] == 0 and len(steps) == 1, "frames must be evenly spaced"
    assert calls[-1] >= 1000 - steps.pop(), "frames must reach the end"
    assert len(draws) < 200, "only the kept steps should be drawn"


def test_recorder_follows_the_first_animated_figure():
    first, second = _small_figure(), _small_figure()
    rec = gifs.FrameRecorder(max_frames=10, width=60)
    rec.offer(first)
    rec.offer(second)
    rec.offer(first)
    plt.close("all")
    assert rec.figure is first
    assert [call for call, _ in rec.frames] == [0, 1]


def test_gif_is_animated_and_holds_the_last_frame(tmp_path):
    frames = [Image.new("RGB", (40, 30), (i * 30, 60, 200 - i * 20))
              for i in range(8)]
    path = tmp_path / "demo.gif"
    gifs.write_gif(frames, path)
    with Image.open(path) as im:
        assert im.format == "GIF" and im.n_frames == 8
        durations = []
        for i in range(im.n_frames):
            im.seek(i)
            durations.append(im.info["duration"])
    assert durations[-1] == gifs.FINAL_HOLD_MS
    assert max(durations[:-1]) < durations[-1]


def test_record_keeps_a_frame_per_pause_and_restores_pyplot():
    before = plt.pause, plt.show

    def demo():
        _small_figure()
        for i in range(30):
            plt.plot([i % 10], [i % 10], "xr")
            plt.pause(0.01)
        plt.show()

    frames = gifs.record(demo, max_frames=10, width=60)
    assert 10 <= len(frames) <= 11  # 10 kept steps (+ the final picture)
    assert (plt.pause, plt.show) == before


def test_record_handles_a_demo_that_closes_its_figure():
    def demo():
        _small_figure()
        for i in range(5):
            plt.plot([i], [i], "xr")
            plt.pause(0.01)
        plt.close()  # e.g. before drawing extra summary plots
        plt.subplots(1)
        plt.show()

    frames = gifs.record(demo, max_frames=10, width=60)
    assert len(frames) >= 5


def test_recorder_follows_a_new_figure_after_the_old_one_closes():
    def demo():  # e.g. coverage planning: one window per example area
        for _ in range(2):
            _small_figure()
            for i in range(4):
                plt.plot([i], [i], "xr")
                plt.pause(0.01)
            plt.close()

    frames = gifs.record(demo, max_frames=20, width=60)
    assert len(frames) >= 8


def test_3d_zoom_makes_the_plot_bigger():
    def demo():
        ax = plt.figure(figsize=(2, 2), dpi=40).add_subplot(projection="3d")
        ax.plot([0, 1], [0, 1], [0, 1], "-r", linewidth=3)
        plt.pause(0.01)

    def red_line_area(frames):
        px = np.asarray(frames[-1]).astype(int)
        red = (px[..., 0] > 180) & (px[..., 1] < 100) & (px[..., 2] < 100)
        rows, cols = np.nonzero(red)
        return (rows.max() - rows.min()) * (cols.max() - cols.min())

    plain = gifs.record(demo, max_frames=5, width=100)
    zoomed = gifs.record(demo, max_frames=5, width=100, zoom3d=1.5)
    assert red_line_area(zoomed) > 1.5 * red_line_area(plain)


def test_record_restores_pyplot_after_an_error():
    before = plt.pause, plt.show

    def demo():
        raise RuntimeError("demo failed")

    with pytest.raises(RuntimeError):
        gifs.record(demo)
    assert (plt.pause, plt.show) == before


def test_record_plays_a_funcanimation():
    def demo():
        fig, ax = plt.subplots(figsize=(2, 1.5), dpi=40)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        line, = ax.plot([], [], "-b")

        def update(i):
            line.set_data(range(i + 1), range(i + 1))
            return line,

        anims = [animation.FuncAnimation(fig, update, frames=20, interval=50,
                                         repeat=False)]
        plt.show()
        return anims

    frames = gifs.record(demo, max_frames=10, width=60)
    assert len(frames) >= 10


def test_last_frame_keeps_blitted_artists():
    """FuncAnimation(blit=True) marks its artists 'animated', and a normal
    draw skips those; the finished picture must still show them."""
    def demo():
        fig, ax = plt.subplots(figsize=(2, 1.5), dpi=40)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        line, = ax.plot([], [], "-k", linewidth=4)

        def update(i):
            line.set_data([0, i], [0, i])
            return line,

        anims = [animation.FuncAnimation(fig, update, frames=20, blit=True,
                                         interval=50, repeat=False)]
        plt.show()
        return anims

    frames = gifs.record(demo, max_frames=10, width=60)
    ink = [int(np.sum(np.asarray(f.convert("L")) < 100)) for f in frames]
    assert ink[-1] >= max(ink), "the last frame lost the animated line"


def test_gif_trims_blank_margins(tmp_path):
    frames = []
    for i in range(3):
        frame = Image.new("RGB", (200, 150), "white")
        frame.paste((0, 0, 0), (50 + i, 40, 70 + i, 60))
        frames.append(frame)
    path = tmp_path / "trimmed.gif"
    gifs.write_gif(frames, path, progress_bar=False)
    with Image.open(path) as im:
        width, height = im.size
    # 22 x 20 pixels of content plus padding, scaled up by at most
    # MAX_UPSCALE: far smaller than the 200 x 150 frame
    assert width < 70 and height < 70, (width, height)


def _bordered_frames(n, size=(640, 480)):
    frames = []
    for i in range(n):
        frame = Image.new("RGB", size, "white")
        frame.paste((120, 120, 120), (0, 0, size[0], 2))  # content reaches
        frame.paste((120, 120, 120), (0, size[1] - 2, size[0], size[1]))
        frame.paste((200, 30, 30), (100 + 40 * i, 200, 140 + 40 * i, 240))
        frames.append(frame)
    return frames


def test_title_strip_and_progress_bar(tmp_path):
    path = tmp_path / "titled.gif"
    gifs.write_gif(_bordered_frames(5), path, title="A* search",
                   section="Path planning")
    with Image.open(path) as im:
        assert im.size[0] == gifs.WIDTH
        im.seek(0)
        first = im.convert("RGB")
        im.seek(im.n_frames - 1)
        last = im.convert("RGB")
    width, height = last.size
    strip = np.asarray(first.crop((0, 0, width, 6)).convert("L"))
    assert strip.mean() < 80, "the title strip should be dark"

    def accent(image):
        px = np.asarray(image.crop((0, height - 3, width, height))).astype(int)
        return np.mean(px[..., 2] - px[..., 0] > 60)

    assert accent(last) > 0.95, "the progress bar ends full"
    assert accent(first) < 0.1, "the progress bar starts (nearly) empty"


def test_frames_are_scaled_to_the_gif_width(tmp_path):
    path = tmp_path / "wide.gif"
    gifs.write_gif(_bordered_frames(3, size=(1280, 960)), path,
                   progress_bar=False)
    with Image.open(path) as im:
        assert im.size == (gifs.WIDTH, gifs.WIDTH * 3 // 4)


def test_crossfade_blends_between_frames(tmp_path):
    frames = [Image.new("RGB", (64, 48), c)
              for c in ((0, 0, 0), (250, 250, 250))]
    path = tmp_path / "fade.gif"
    gifs.write_gif(frames, path, crossfade=4, progress_bar=False)
    with Image.open(path) as im:
        assert im.n_frames == 2 + 4
        im.seek(3)
        middle = np.asarray(im.convert("L")).mean()
    assert 60 < middle < 200, "in-between frames should be blends"


def test_a_finished_plot_can_get_a_better_figure_size():
    def demo():
        plt.figure(figsize=(4, 1.5), dpi=40)  # wide and flat
        plt.plot([0, 1], [0, 1], "-r")
        plt.show()

    frames = gifs.record(demo, max_frames=10, width=60, figsize=(2, 2))
    assert frames[-1].size[0] == frames[-1].size[1], "square, as asked"


def test_a_finished_plot_is_replayed_in_drawing_order():
    def demo():
        _small_figure()
        line = np.linspace(0, 10, 200)
        plt.plot(line, line, "-r")  # long line: grows point by point
        plt.plot([5], [2], "ob")
        plt.show()

    frames = gifs.record(demo, max_frames=30, width=60)
    ink = [int(np.sum(np.asarray(f.convert("L")) < 200)) for f in frames]
    assert len(frames) >= 5
    assert ink[0] < ink[len(ink) // 2] < ink[-1], "the plot should build up"


def _gifs_in_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return {pathlib.Path(p).stem
            for p in re.findall(r"\]\((gifs/[\w.-]+\.gif)\)", text)}


def test_every_readme_gif_can_be_made_again():
    assert _gifs_in_readme() == set(gifs.DEMOS)


@pytest.mark.parametrize("name", sorted(gifs.DEMOS))
def test_demo_recipe_points_at_real_code(name):
    demo = gifs.DEMOS[name]
    assert demo.title and demo.section, "every GIF needs a title strip"
    spec = importlib.util.find_spec(demo.module)
    assert spec is not None, f"{demo.module} cannot be imported"
    if isinstance(demo.entry, str):
        tree = ast.parse(pathlib.Path(spec.origin).read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert demo.entry in defined, f"{demo.module} has no {demo.entry}()"


if __name__ == '__main__':
    conftest.run_this_test(__file__)
