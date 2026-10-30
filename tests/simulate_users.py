"""
Simulate many users sending planning requests to the grid planners at once,
and check every answer.

Each simulated user picks A* or Dijkstra, a random map (random size, origin,
resolution and robot radius), a start and a goal. Some requests are
deliberately broken (invalid input, start off the map, unreachable goal,
start == goal). Requests run in a pool of worker processes, so each worker
also handles many requests one after another. Every answer is checked
against an independent shortest-path search.

Optionally (--compare-old DIR) the same request is also sent to an older
copy of the planners, to confirm both give exactly the same path.

Usage:
    python tests/simulate_users.py --users 5000 --workers 2

author: Mahim Trivedi
"""
import argparse
import collections
import importlib.util
import math
import pathlib
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import grid_planner_checks as chk  # noqa: E402
from PathPlanning.AStar import a_star  # noqa: E402
from PathPlanning.Dijkstra import dijkstra  # noqa: E402

a_star.show_animation = False
dijkstra.show_animation = False
PLANNERS = {"astar": a_star.AStarPlanner, "dijkstra": dijkstra.DijkstraPlanner}
OLD = {}


def load_old(base_dir):
    for kind, rel, cls in (("astar", "PathPlanning/AStar/a_star.py", "AStarPlanner"),
                           ("dijkstra", "PathPlanning/Dijkstra/dijkstra.py",
                            "DijkstraPlanner")):
        spec = importlib.util.spec_from_file_location(f"old_{kind}",
                                                      pathlib.Path(base_dir) / rel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.show_animation = False
        OLD[kind] = getattr(module, cls)


def make_request(user_id):
    rng = random.Random(user_id)
    request = {"user": user_id, "planner": rng.choice(sorted(PLANNERS)),
               "case": "normal", "resolution": rng.choice([1.0, 1.0, 2.0, 1.5]),
               "rr": rng.choice([0.0, 0.5, 1.0])}
    ox, oy, start, goal = chk.random_scenario(user_id, size=rng.randint(12, 30),
                                              shift="random")
    roll = rng.random()
    if roll < 0.03:
        request["case"] = "invalid"
        bad = rng.choice(["resolution", "nan_obstacle", "empty", "nan_start"])
        if bad == "resolution":
            request["resolution"] = rng.choice([0.0, -1.0, float("nan")])
        elif bad == "nan_obstacle":
            ox = ox[:5] + [float("nan")] + ox[5:]
        elif bad == "empty":
            ox, oy = [], []
        else:
            start = (float("nan"), start[1])
    elif roll < 0.08:
        request["case"] = "off_map"
        start = (min(ox) - 20.0, start[1])
    elif roll < 0.13:
        request["case"] = "start_is_goal"
        goal = start
    elif roll < 0.18:
        request["case"] = "unreachable"
        # wall the goal in with a box of obstacle points; 1 m grid and 1 m
        # robot radius make the box solid (point walls leak on coarser grids)
        request.update(resolution=1.0, rr=1.0)
        gx, gy = goal
        for d in range(-2, 3):
            ox += [gx + d, gx + d, gx - 2, gx + 2]
            oy += [gy - 2, gy + 2, gy + d, gy + d]
    request.update(ox=ox, oy=oy, start=start, goal=goal)
    return request


def handle(user_id):
    request = make_request(user_id)
    cls = PLANNERS[request["planner"]]
    result = {"user": user_id, "planner": request["planner"],
              "case": request["case"]}
    t0 = time.perf_counter()
    try:
        planner = cls(request["ox"], request["oy"], request["resolution"],
                      request["rr"])
        t1 = time.perf_counter()
        rx, ry = planner.planning(*request["start"], *request["goal"])
        t2 = time.perf_counter()
    except ValueError as err:
        result["status"] = "ok" if request["case"] == "invalid" else "wrong"
        result["detail"] = f"rejected: {err}"
        return result
    except Exception as err:  # any other exception is a crash
        result["status"] = "crash"
        result["detail"] = repr(err)
        return result
    result["build_ms"] = (t1 - t0) * 1e3
    result["plan_ms"] = (t2 - t1) * 1e3
    try:
        if request["case"] == "invalid":
            raise AssertionError("invalid input was accepted")
        if request["case"] == "off_map":
            assert (rx, ry) == ([], []), "start off the map must give no path"
        elif request["case"] == "start_is_goal":
            assert len(rx) == 1, "start == goal must give a one-point path"
        else:
            expected = chk.reference_cost(planner, *request["start"],
                                          *request["goal"])
            if expected is None:
                assert (rx, ry) == ([], []), "unreachable goal must be empty"
            else:
                chk.assert_valid_path(planner, rx, ry, request["start"],
                                      request["goal"])
                cost = chk.path_cost(planner, rx, ry)
                assert math.isclose(cost, expected, abs_tol=1e-9), \
                    f"cost {cost} but shortest is {expected}"
                chk.assert_clear_of_raw_obstacles(planner, rx, ry,
                                                  request["ox"], request["oy"])
        result["status"] = "ok"
        result["empty"] = rx == []
    except AssertionError as err:
        result["status"] = "wrong"
        result["detail"] = str(err)
        return result

    if OLD and request["case"] == "normal" and rx and \
            min(request["ox"]) <= 0 and min(request["oy"]) <= 0:
        # old planners are only trusted on maps with the origin inside them
        try:
            old = OLD[request["planner"]](request["ox"], request["oy"],
                                          request["resolution"], request["rr"])
            result["same_as_old"] = old.planning(*request["start"],
                                                 *request["goal"]) == (rx, ry)
        except Exception:
            result["same_as_old"] = None  # the old version crashed
    return result


def init_worker(base_dir):
    import io
    import contextlib
    sys.stdout = io.StringIO()  # the planners print progress lines
    if base_dir:
        with contextlib.redirect_stdout(io.StringIO()):
            load_old(base_dir)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--compare-old", default="")
    args = parser.parse_args()

    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                             initargs=(args.compare_old,)) as pool:
        results = list(pool.map(handle, range(args.users), chunksize=25))
    elapsed = time.perf_counter() - start

    status = collections.Counter(r["status"] for r in results)
    print(f"simulated users: {args.users} in {elapsed:.1f} s "
          f"({args.users / elapsed:.0f} requests/s, {args.workers} workers)")
    print("results:", dict(status))
    by_case = collections.defaultdict(collections.Counter)
    for r in results:
        by_case[r["case"]][r["status"]] += 1
    for case, counts in sorted(by_case.items()):
        print(f"  {case:14s} {dict(counts)}")
    for name in ("build_ms", "plan_ms"):
        values = [r[name] for r in results if name in r]
        print(f"{name}: median {statistics.median(values):.1f}, "
              f"p95 {percentile(values, 0.95):.1f}, "
              f"p99 {percentile(values, 0.99):.1f}, max {max(values):.1f}")
    walled = [r for r in results if r["case"] == "unreachable" and "empty" in r]
    print(f"walled-in goals answered with an empty path: "
          f"{sum(r['empty'] for r in walled)}/{len(walled)} "
          f"(the rest had the start inside the box; the reference agreed)")
    same = [r["same_as_old"] for r in results if "same_as_old" in r]
    if same:
        print(f"old vs new on {len(same)} requests: identical "
              f"{same.count(True)}, different {same.count(False)}, "
              f"old crashed {same.count(None)}")
    problems = [r for r in results if r["status"] != "ok"]
    for r in problems[:10]:
        print("PROBLEM", r)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
