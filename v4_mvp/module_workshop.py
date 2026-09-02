from __future__ import annotations

import ast
import csv
import difflib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODULE_STORE_PATH = Path(os.environ.get(
    "LEESIN_V4_MODULE_STORE",
    str(Path(__file__).resolve().parent / "runtime" / "custom_modules.json"),
))
_LOCK = threading.RLock()
MAX_CODE_CHARS = 50_000
MAX_TABLE_ROWS = 10_000
RUN_TIMEOUT_SECONDS = 3.0

SAFE_LIBRARY_ATTRS = {
    "math": {"ceil", "cos", "exp", "floor", "fsum", "log", "log10", "log2", "pi", "pow", "sin", "sqrt", "tan"},
    "statistics": {"fmean", "mean", "median", "mode", "multimode", "pstdev", "pvariance", "quantiles", "stdev", "variance"},
}
SAFE_METHODS = {"append", "count", "endswith", "extend", "get", "items", "join", "keys", "lower", "pop", "replace", "sort", "split", "startswith", "strip", "upper", "values"}


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in SAFE_LIBRARY_ATTRS:
        raise ImportError("Only math/statistics imports are enabled in the MVP runner.")
    if fromlist:
        for item in fromlist:
            if item == "*" or item not in SAFE_LIBRARY_ATTRS[name]:
                raise ImportError(f"from {name} import {item} is not enabled.")
    return math if name == "math" else statistics


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError, "__import__": _safe_import,
}


def _parse_code(code: str) -> ast.Module:
    code = str(code or "")
    if not code.strip():
        raise ValueError("Python function code is required.")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError("Code is too large for the MVP runner.")
    try:
        return ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error at line {exc.lineno}: {exc.msg}") from exc


def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _params(fn: ast.FunctionDef) -> list[dict[str, Any]]:
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    defaults = [None] * (len(positional) - len(fn.args.defaults)) + list(fn.args.defaults)
    posonly = {arg.arg for arg in fn.args.posonlyargs}
    out = []
    for arg, default in zip(positional, defaults):
        out.append({
            "name": arg.arg,
            "kind": "positional_only" if arg.arg in posonly else "positional_or_keyword",
            "required": default is None,
            "default": None if default is None else ast.unparse(default),
            "annotation": None if arg.annotation is None else ast.unparse(arg.annotation),
        })
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        out.append({
            "name": arg.arg, "kind": "keyword_only", "required": default is None,
            "default": None if default is None else ast.unparse(default),
            "annotation": None if arg.annotation is None else ast.unparse(arg.annotation),
        })
    if fn.args.vararg:
        out.append({"name": fn.args.vararg.arg, "kind": "var_positional", "required": False, "default": None, "annotation": None})
    if fn.args.kwarg:
        out.append({"name": fn.args.kwarg.arg, "kind": "var_keyword", "required": False, "default": None, "annotation": None})
    return out


def inspect_python(code: str) -> dict[str, Any]:
    functions = _functions(_parse_code(code))
    if not functions:
        raise ValueError("Paste at least one top-level `def ...` function.")
    items = [{"name": fn.name, "parameters": _params(fn), "docstring": ast.get_docstring(fn) or "", "line": fn.lineno} for fn in functions]
    recommended = next((item["name"] for item in items if not item["name"].startswith("_")), items[0]["name"])
    return {"functions": items, "recommendedFunction": recommended}


def _typed(value: Any) -> Any:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def parse_table(data_text: str) -> dict[str, Any]:
    text = str(data_text or "").lstrip("\ufeff")
    if not text.strip():
        raise ValueError("Paste CSV/TSV/Excel table data first.")
    first = next((line for line in text.splitlines() if line.strip()), "")
    delimiters = ["\t", ",", ";", "|"]
    counts = {d: first.count(d) for d in delimiters}
    delimiter = max(delimiters, key=lambda d: counts[d]) if max(counts.values()) else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = [str(c or "").strip() for c in (reader.fieldnames or [])]
    if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Data needs a unique, non-empty header row.")
    rows = []
    for raw in reader:
        if len(rows) >= MAX_TABLE_ROWS:
            raise ValueError(f"The MVP accepts at most {MAX_TABLE_ROWS:,} rows per paste.")
        row = {c: _typed(raw.get(c)) for c in columns}
        if any(v is not None for v in row.values()):
            rows.append(row)
    if not rows:
        raise ValueError("The pasted table has no data rows.")
    return {"columns": columns, "rows": rows, "rowCount": len(rows), "preview": rows[:8], "delimiter": "TAB" if delimiter == "\t" else delimiter}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def suggest_mapping(meta: dict[str, Any], columns: list[str]) -> dict[str, str]:
    mapping, used = {}, set()
    aliases = {"runtime": {"runtimems", "elapsed", "elapsedms", "timems", "time"}, "n": {"n", "size", "inputsize"}, "x": {"x", "input", "feature"}, "y": {"y", "output", "target", "score"}}
    for param in meta.get("parameters", []):
        name, kind = param["name"], param["kind"]
        if kind in {"var_positional", "var_keyword"}:
            continue
        p = _norm(name)
        if p in {"rows", "records", "data", "dataset", "table"}:
            mapping[name] = "__rows__"
            continue
        ranked = []
        for column in columns:
            if column in used:
                continue
            c = _norm(column)
            score = 1.0 if p == c else 0.9 if c in aliases.get(p, set()) else 0.86 if p in c or c in p else difflib.SequenceMatcher(a=p, b=c).ratio() * 0.78
            ranked.append((score, column))
        ranked.sort(reverse=True)
        if ranked and ranked[0][0] >= 0.58:
            mapping[name] = ranked[0][1]
            used.add(ranked[0][1])
        elif not param.get("required"):
            mapping[name] = "__default__"
    return mapping


def prepare_workshop(code: str, data_text: str, function_name: str | None = None) -> dict[str, Any]:
    inspected, table = inspect_python(code), parse_table(data_text)
    name = function_name or inspected["recommendedFunction"]
    selected = next((item for item in inspected["functions"] if item["name"] == name), None)
    if not selected:
        raise ValueError(f"Unknown function: {name}")
    return {
        "functions": inspected["functions"], "selectedFunction": selected,
        "data": {key: table[key] for key in ("columns", "rowCount", "preview", "delimiter")},
        "suggestedMapping": suggest_mapping(selected, table["columns"]),
    }


class _Validator(ast.NodeVisitor):
    BLOCKED = (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)

    def __init__(self, function_names: set[str], library_aliases: dict[str, str], direct_imports: set[str]):
        self.function_names, self.library_aliases, self.direct_imports = function_names, library_aliases, direct_imports
        self.errors: list[str] = []

    def visit(self, node: ast.AST):
        if isinstance(node, self.BLOCKED):
            self.errors.append(f"{type(node).__name__} is not enabled.")
            return None
        return super().visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in SAFE_LIBRARY_ATTRS:
                self.errors.append(f"Import `{alias.name}` is not enabled.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level or node.module not in SAFE_LIBRARY_ATTRS:
            self.errors.append(f"Import from `{node.module}` is not enabled.")
            return
        for alias in node.names:
            if alias.name == "*" or alias.name not in SAFE_LIBRARY_ATTRS[node.module]:
                self.errors.append(f"from {node.module} import {alias.name} is not enabled.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id.startswith("__"):
            self.errors.append(f"Name `{node.id}` is not enabled.")

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("_"):
            self.errors.append(f"Attribute `{node.attr}` is not enabled.")
            return
        if isinstance(node.value, ast.Name) and node.value.id in self.library_aliases:
            lib = self.library_aliases[node.value.id]
            if node.attr not in SAFE_LIBRARY_ATTRS[lib]:
                self.errors.append(f"{node.value.id}.{node.attr} is not enabled.")
        elif node.attr not in SAFE_METHODS:
            self.errors.append(f"Method `.{node.attr}` is not enabled.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in SAFE_BUILTINS and name not in self.function_names and name not in self.direct_imports:
                self.errors.append(f"Call to `{name}` is not enabled.")
                return
        self.generic_visit(node)


def validate_runnable_code(code: str) -> None:
    tree = _parse_code(code)
    functions = _functions(tree)
    if not functions:
        raise ValueError("Paste at least one top-level function.")
    aliases = {"math": "math", "statistics": "statistics"}
    direct = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in SAFE_LIBRARY_ATTRS:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in SAFE_LIBRARY_ATTRS:
            for alias in node.names:
                if alias.name in SAFE_LIBRARY_ATTRS[node.module]:
                    direct.add(alias.asname or alias.name)
    validator = _Validator({fn.name for fn in functions}, aliases, direct)
    validator.visit(tree)
    if validator.errors:
        raise ValueError("Code is outside the MVP runner subset: " + " ".join(dict.fromkeys(validator.errors)))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return str(value) if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _worker_main() -> int:
    payload = json.loads(sys.stdin.read())
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "math": math, "statistics": statistics}
    started = time.perf_counter()
    try:
        exec(compile(payload["code"], "<leesin-module>", "exec"), namespace, namespace)
        fn = namespace.get(payload["functionName"])
        if not callable(fn):
            raise ValueError("Entry function was not created.")
        result = fn(*(payload.get("args") or []), **(payload.get("kwargs") or {}))
        out = {"ok": True, "result": _json_safe(result), "resultType": type(result).__name__}
    except Exception as exc:
        out = {"ok": False, "error": str(exc), "errorType": type(exc).__name__}
    out["executionMs"] = (time.perf_counter() - started) * 1000
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


def _meta(code: str, name: str) -> dict[str, Any]:
    inspected = inspect_python(code)
    found = next((item for item in inspected["functions"] if item["name"] == name), None)
    if not found:
        raise ValueError(f"Unknown function: {name}")
    return found


def run_workshop(code: str, function_name: str, data_text: str, mapping: dict[str, Any]) -> dict[str, Any]:
    validate_runnable_code(code)
    meta, table = _meta(code, function_name), parse_table(data_text)
    args, kwargs = [], {}
    for param in meta["parameters"]:
        if param["kind"] in {"var_positional", "var_keyword"}:
            continue
        source = str(mapping.get(param["name"]) or "")
        if source == "__default__" or (not source and not param["required"]):
            continue
        if not source:
            raise ValueError(f"Map required input `{param['name']}`.")
        value = table["rows"] if source == "__rows__" else [row.get(source) for row in table["rows"]] if source in table["columns"] else None
        if value is None:
            raise ValueError(f"Unknown mapping source: {source}")
        if param["kind"] == "positional_only":
            args.append(value)
        else:
            kwargs[param["name"]] = value
    payload = json.dumps({"code": code, "functionName": function_name, "args": args, "kwargs": kwargs}, ensure_ascii=False)
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
            input=payload, text=True, capture_output=True, timeout=RUN_TIMEOUT_SECONDS,
            env={"PYTHONIOENCODING": "utf-8"}, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Module execution exceeded 3 seconds and was stopped.") from exc
    if completed.returncode:
        raise ValueError(completed.stderr.strip() or "Worker process failed.")
    output = json.loads(completed.stdout)
    if not output.get("ok"):
        raise ValueError(f"{output.get('errorType')}: {output.get('error')}")
    return {"function": function_name, "mapping": mapping, "rowCount": table["rowCount"], "result": output["result"], "resultType": output["resultType"], "executionMs": output["executionMs"]}


def _read_store() -> dict[str, Any]:
    with _LOCK:
        if not MODULE_STORE_PATH.exists():
            return {"modules": {}}
        try:
            data = json.loads(MODULE_STORE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"modules": {}}
        if not isinstance(data.get("modules"), dict):
            data["modules"] = {}
        return data


def _write_store(data: dict[str, Any]) -> None:
    MODULE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="modules-", suffix=".json", dir=str(MODULE_STORE_PATH.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, MODULE_STORE_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def list_saved_modules() -> list[dict[str, Any]]:
    modules = [dict(item) for item in _read_store()["modules"].values()]
    return sorted(modules, key=lambda item: item.get("createdAt", ""), reverse=True)


def save_module(*, code: str, function_name: str, title: str = "", description: str = "", question: str = "", assumptions: str = "", limits: str = "") -> dict[str, Any]:
    validate_runnable_code(code)
    meta = _meta(code, function_name)
    with _LOCK:
        data = _read_store()
        module_id = f"module_{uuid.uuid4().hex[:10]}"
        item = {
            "id": module_id, "title": title.strip() or function_name,
            "description": description.strip() or meta["docstring"], "question": question.strip(),
            "assumptions": [line.strip() for line in assumptions.splitlines() if line.strip()],
            "limits": [line.strip() for line in limits.splitlines() if line.strip()],
            "entryFunction": function_name, "inputContract": meta["parameters"], "code": code,
            "version": "0.1.0", "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        data["modules"][module_id] = item
        _write_store(data)
        return dict(item)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    raise SystemExit("Use `python -m v4_mvp.app` to run Leesin_V4.")
