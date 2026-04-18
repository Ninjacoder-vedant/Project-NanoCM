import re
import ast
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

GENERATED_TESTS_DIR = Path("hf_cache/generated_tests")

def _coerce_tests(val) -> list[dict]:
    """Normalize official_tests (or similar) into a plain list of dicts."""
    if val is None:
        return []
    # numpy array of dicts (common with parquet)
    if isinstance(val, np.ndarray):
        return val.tolist()
    # already a list
    if isinstance(val, list):
        return val
    # string representation
    if isinstance(val, str):
        return ast.literal_eval(val)
    # pandas can also produce these
    try:
        return list(val)
    except TypeError:
        return []

# ---------- Load generated tests for a problem ----------
@lru_cache(maxsize=128)
def _load_contest_parquet(contest_id: int) -> pd.DataFrame | None:
    """Cache parquet files per contest (many problems share one file)."""
    path = GENERATED_TESTS_DIR / f"test_cases_{contest_id:04d}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

def load_generated_tests(problem_id: str, contest_id: int) -> list[dict]:
    """Return list of {'input': ..., 'output': ...} for the given problem."""
    df = _load_contest_parquet(contest_id)
    if df is None:
        return []
    sub = df[df["problem_id"] == problem_id]
    return [{"input": r["input"], "output": r["output"]} for _, r in sub.iterrows()]


# ---------- Extract, compile, run, check (same as before) ----------
CODE_RE = re.compile(r"```cpp\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_cpp(response: str) -> str | None:
    matches = CODE_RE.findall(response)
    if not matches:
        m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        return m.group(1) if m else None
    return matches[-1]

def compile_cpp(code: str, workdir: Path) -> tuple[Path | None, str]:
    src = workdir / "sol.cpp"
    exe = workdir / "sol"
    src.write_text(code)
    proc = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", str(exe), str(src)],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return None, proc.stderr
    return exe, ""

def run_binary(exe: Path, stdin_data: str, time_limit: float) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            [str(exe)], input=stdin_data, capture_output=True, text=True,
            timeout=time_limit + 0.5,
        )
    except subprocess.TimeoutExpired:
        return "tle", ""
    if proc.returncode != 0:
        return "rte", proc.stdout
    return "ok", proc.stdout

def normalize(s: str) -> list[list[str]]:
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return [ln.split() for ln in lines]

def direct_match(expected: str, actual: str) -> bool:
    return normalize(expected) == normalize(actual)

def checker_match(checker_src: str, inp: str, ref_out: str, sub_out: str,
                  workdir: Path) -> bool:
    chk = workdir / "checker.py"; chk.write_text(checker_src)
    inp_f = workdir / "in.txt";   inp_f.write_text(inp)
    ref_f = workdir / "ref.txt";  ref_f.write_text(ref_out)
    sub_f = workdir / "sub.txt";  sub_f.write_text(sub_out)
    try:
        proc = subprocess.run(
            ["python3", str(chk), str(inp_f), str(ref_f), str(sub_f)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.stdout.strip() == "1"


# ---------- Run one set of tests against a compiled binary ----------
def run_tests(exe: Path, tests: list[dict], time_limit: float,
              checker_src: str | None, workdir: Path) -> dict:
    passed, total = 0, len(tests)
    per_test = []
    for t in tests:
        status, out = run_binary(exe, t["input"], time_limit)
        if status == "tle":
            per_test.append("tle"); continue
        if status == "rte":
            per_test.append("rte"); continue
        if checker_src:
            ok = checker_match(checker_src, t["input"], t["output"], out, workdir)
        else:
            ok = direct_match(t["output"], out)
        if ok:
            passed += 1
            per_test.append("ok")
        else:
            per_test.append("wa")
    return {"passed": passed, "total": total, "per_test": per_test}


# ---------- Evaluate one problem on official + generated ----------
def evaluate_problem(row: dict, llm_response: str) -> dict:
    result = {
        "id": row["id"],
        "compile_ok": False, "error": None,
        "official_passed": 0, "official_total": 0, "official_accuracy": 0.0,
        "generated_passed": 0, "generated_total": 0, "generated_accuracy": 0.0,
        "total_passed": 0, "total_total": 0, "total_accuracy": 0.0,
        "official_per_test": [], "generated_per_test": [],
    }

    code = extract_cpp(llm_response)
    if code is None:
        result["error"] = "no_code_extracted"
        return result

    # Official tests
    official = _coerce_tests(row.get("official_tests"))

    # Generated tests
    generated = load_generated_tests(row["id"], int(row["contest_id"]))

    if not official and not generated:
        result["error"] = "no_tests"
        return result

    time_limit = float(row.get("time_limit") or 1.0)
    checker_src = row.get("generated_checker")

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        exe, err = compile_cpp(code, workdir)
        if exe is None:
            result["error"] = f"compile_error: {err[:500]}"
            return result
        result["compile_ok"] = True

        if official:
            r = run_tests(exe, official, time_limit, checker_src, workdir)
            result["official_passed"] = r["passed"]
            result["official_total"]  = r["total"]
            result["official_per_test"] = r["per_test"]
            result["official_accuracy"] = r["passed"] / r["total"] if r["total"] else 0.0

        if generated:
            r = run_tests(exe, generated, time_limit, checker_src, workdir)
            result["generated_passed"] = r["passed"]
            result["generated_total"]  = r["total"]
            result["generated_per_test"] = r["per_test"]
            result["generated_accuracy"] = r["passed"] / r["total"] if r["total"] else 0.0

    result["total_passed"] = result["official_passed"] + result["generated_passed"]
    result["total_total"]  = result["official_total"]  + result["generated_total"]
    if result["total_total"]:
        result["total_accuracy"] = result["total_passed"] / result["total_total"]
    return result


# ---------- Parallel driver ----------
def _worker(args):
    row_dict, llm_response = args
    return evaluate_problem(row_dict, llm_response)

def evaluate_dataframe(df, response_col="llm_response", n_workers=4):
    tasks = [(row, row[response_col]) for row in df.to_dict("records")]
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    df = df.copy()
    for key in ["compile_ok", "error",
                "official_passed", "official_total", "official_accuracy",
                "generated_passed", "generated_total", "generated_accuracy",
                "total_passed", "total_total", "total_accuracy",
                "official_per_test", "generated_per_test"]:
        df[key] = [r[key] for r in results]
    return df