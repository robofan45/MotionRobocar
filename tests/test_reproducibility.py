"""
Every demo that draws random numbers must seed its random generators,
so each run gives the same result.

This is a static (AST based) check, so it is fast and covers every module.

author: Mahim Trivedi

Rules, for each entry function (``main*`` functions and functions called from
the ``if __name__ == '__main__':`` block):
- if it can reach draws from the legacy ``np.random.*`` functions (directly or
  through functions and classes of the same module) it must call
  ``np.random.seed(...)``; if it can reach draws from the ``random`` module
  (also via ``from random import ...``) it must call ``random.seed(...)``;
- any generator the entry function creates (``default_rng``, ``RandomState``,
  ``random.Random``) must get a seed.
Library functions may stay random; the demo entry point decides the seed.

Limits: randomness reached only through another module, or behaviour that
follows the wall clock, is not visible here. The per-demo tests that run a
demo twice and compare the results cover those.
"""
import ast
import pathlib

import conftest  # noqa: F401  (adds the project root to sys.path)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_TOP_DIRS = {"tests", "docs"}

NP_SEED = {"np.random.seed", "numpy.random.seed"}
PY_SEED = {"random.seed"}
NOT_A_DRAW = {"np.random.get_state", "np.random.set_state",
              "random.getstate", "random.setstate"}
GENERATORS = ("default_rng", "RandomState", "Random")


def _modules():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in SKIP_TOP_DIRS:
            continue
        yield path, rel.as_posix()


def _call_name(call):
    return ast.unparse(call.func)


def _creates_generator(name):
    return name.split(".")[-1] in GENERATORS


def _uses_generator_object(name):
    """e.g. np.random.default_rng(1).random -> a dedicated generator."""
    return any(f"{g}(" in name for g in GENERATORS)


def _from_random_names(tree):
    return {a.asname or a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module == "random"
            for a in n.names}


def _imports_random(tree):
    return any(
        isinstance(n, ast.Import) and any(a.name == "random" for a in n.names)
        for n in ast.walk(tree))


def _draw_kinds(node, imports_random, from_random):
    """Random generators drawn from anywhere inside ``node``."""
    kinds = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        name = _call_name(sub)
        if (name in NOT_A_DRAW or name in NP_SEED or name in PY_SEED
                or _creates_generator(name) or _uses_generator_object(name)):
            continue
        if name.startswith(("np.random.", "numpy.random.")):
            kinds.add("np.random")
        elif imports_random and name.startswith("random."):
            kinds.add("random")
        elif name in from_random and name != "seed":
            kinds.add("random")
    return kinds


def _reachable_draw_kinds(tree):
    """For each top-level function: the generators it can reach by calling
    functions and classes defined in the same module."""
    imports_random, from_random = _imports_random(tree), _from_random_names(tree)
    defs = {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    memo = {}

    def reach(name, seen):
        if name in memo:
            return memo[name]
        if name in seen or name not in defs:
            return set()
        node = defs[name]
        kinds = _draw_kinds(node, imports_random, from_random)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                kinds |= reach(sub.func.id, seen | {name})
        memo[name] = kinds
        return kinds

    return {name: reach(name, set()) for name in defs}


def _entry_functions(tree):
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    entries = {name for name in funcs if name.startswith("main")}
    for node in tree.body:
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id in funcs):
                    entries.add(sub.func.id)
    return [funcs[name] for name in sorted(entries)]


def _seeded_kinds(func, from_random):
    kinds = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in NP_SEED:
                kinds.add("np.random")
            elif name in PY_SEED or (name == "seed" and "seed" in from_random):
                kinds.add("random")
    return kinds


def test_entry_functions_seed_random_generators():
    problems = []
    for path, rel in _modules():
        tree = ast.parse(path.read_text("utf-8"))
        reachable = _reachable_draw_kinds(tree)
        from_random = _from_random_names(tree)
        for func in _entry_functions(tree):
            missing = reachable[func.name] - _seeded_kinds(func, from_random)
            if missing:
                problems.append(
                    f"{rel}: {func.name}() does not seed {sorted(missing)}")
    assert not problems, "\n".join(problems)


def test_entry_functions_create_seeded_generators():
    problems = []
    for path, rel in _modules():
        tree = ast.parse(path.read_text("utf-8"))
        for func in _entry_functions(tree):
            for node in ast.walk(func):
                if (isinstance(node, ast.Call)
                        and _creates_generator(_call_name(node))
                        and not node.args and not node.keywords):
                    problems.append(f"{rel}:{node.lineno}: {func.name}() "
                                    "creates a generator without a seed")
    assert not problems, "\n".join(problems)


if __name__ == '__main__':
    conftest.run_this_test(__file__)
