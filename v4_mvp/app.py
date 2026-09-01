from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from v4_mvp.modules import MODULE_VERSION, analyze_question, question_registry
from v4_mvp.mvp_adapters.prime_benchmark import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    generate_cluster_payload,
)
from v4_mvp.store import (
    add_cluster,
    create_project,
    list_projects,
    project_detail,
    save_analysis,
    selected_clusters,
    start_proposal,
)


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
MVP_PRIME_UI_PATH = Path(__file__).resolve().parent / "mvp_adapters" / "prime_ui.js"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class V4Handler(BaseHTTPRequestHandler):
    server_version = "LeesinV4MVP/0.1"

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        # MVP-only experiment UI is injected here instead of being mixed into the
        # main template. Remove this one line and the /mvp-prime-ui.js route when
        # built-in experiment execution is no longer needed.
        text = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
            "</body>", '<script src="/mvp-prime-ui.js"></script>\n</body>'
        )
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_javascript(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def _segments(self) -> list[str]:
        return [part for part in urlparse(self.path).path.split("/") if part]

    def do_GET(self) -> None:  # noqa: N802
        try:
            segments = self._segments()
            if not segments:
                self._send_html()
                return
            if segments == ["mvp-prime-ui.js"]:
                self._send_javascript(MVP_PRIME_UI_PATH)
                return
            if segments == ["api", "bootstrap"]:
                self._send_json(
                    {
                        "projects": list_projects(),
                        "questions": question_registry(),
                        "moduleVersion": MODULE_VERSION,
                        "mvpExperimentAdapter": {
                            "id": "prime-benchmark-mvp-v1",
                            "repeats": DEFAULT_REPEATS,
                            "warmup": DEFAULT_WARMUP,
                        },
                    }
                )
                return
            if len(segments) == 3 and segments[:2] == ["api", "projects"]:
                self._send_json(project_detail(segments[2]))
                return
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json(
                {"error": str(exc), "errorType": exc.__class__.__name__},
                HTTPStatus.BAD_REQUEST,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            segments = self._segments()
            body = self._read_json()
            if segments == ["api", "projects"]:
                self._send_json(
                    create_project(body.get("title", ""), body.get("description", "")),
                    HTTPStatus.CREATED,
                )
                return
            if (
                len(segments) == 4
                and segments[:2] == ["api", "projects"]
                and segments[3] == "clusters"
            ):
                cluster = add_cluster(
                    segments[2],
                    name=body.get("name", ""),
                    filename=body.get("filename", ""),
                    csv_text=body.get("csvText", ""),
                    protocol=body.get("protocol", ""),
                    context=body.get("context", ""),
                    origin_proposal_id=body.get("originProposalId"),
                )
                self._send_json(cluster, HTTPStatus.CREATED)
                return
            if (
                len(segments) == 5
                and segments[:2] == ["api", "projects"]
                and segments[3:] == ["mvp", "prime-benchmark"]
            ):
                n_values = body.get("nValues") or []
                if not isinstance(n_values, list):
                    raise ValueError("nValues must be a list.")
                payload = generate_cluster_payload(
                    [int(item) for item in n_values],
                    repeats=DEFAULT_REPEATS,
                    warmup=DEFAULT_WARMUP,
                    origin_proposal_id=(
                        str(body.get("originProposalId"))
                        if body.get("originProposalId")
                        else None
                    ),
                )
                cluster = add_cluster(
                    segments[2],
                    name=payload["name"],
                    filename=payload["filename"],
                    csv_text=payload["csv_text"],
                    protocol=payload["protocol"],
                    context=payload["context"],
                    origin_proposal_id=payload["origin_proposal_id"],
                )
                row_count = len(
                    [line for line in payload["csv_text"].splitlines()[1:] if line.strip()]
                )
                self._send_json(
                    {
                        "cluster": cluster,
                        "experiment": payload["experiment"],
                        "rowCount": row_count,
                    },
                    HTTPStatus.CREATED,
                )
                return
            if (
                len(segments) == 4
                and segments[:2] == ["api", "projects"]
                and segments[3] == "analyze"
            ):
                project_id = segments[2]
                question_id = str(body.get("questionId") or "")
                cluster_ids = body.get("clusterIds") or []
                if not isinstance(cluster_ids, list):
                    raise ValueError("clusterIds must be a list.")
                cluster_ids = [str(item) for item in cluster_ids]
                outcome = analyze_question(
                    question_id, selected_clusters(project_id, cluster_ids)
                )
                analysis = save_analysis(
                    project_id,
                    question_id=question_id,
                    cluster_ids=cluster_ids,
                    module_version=MODULE_VERSION,
                    outcome=outcome,
                )
                self._send_json(analysis, HTTPStatus.CREATED)
                return
            if (
                len(segments) == 6
                and segments[:2] == ["api", "projects"]
                and segments[3] == "proposals"
                and segments[5] == "start"
            ):
                self._send_json(start_proposal(segments[2], segments[4]))
                return
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json(
                {"error": str(exc), "errorType": exc.__class__.__name__},
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[Leesin_V4] {self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), V4Handler)
    print(f"Leesin_V4 MVP: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Leesin_V4 MVP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
