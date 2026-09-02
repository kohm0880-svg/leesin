from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from v4_mvp import store
from v4_mvp.workspace_store import (
    add_project_file,
    create_folder,
    get_project_file,
    restore_items,
    trash_items,
    workspace_detail,
)


class WorkspaceStoreTests(unittest.TestCase):
    def setUp(self):
        self._old_store_path = store.STORE_PATH
        self._tmp = tempfile.TemporaryDirectory()
        store.STORE_PATH = Path(self._tmp.name) / "store.json"
        self.project = store.create_project("Workspace test")

    def tearDown(self):
        store.STORE_PATH = self._old_store_path
        self._tmp.cleanup()

    def test_original_file_bytes_are_preserved_and_restorable(self):
        folder = create_folder(self.project["id"], name="raw")
        raw = b"a,b\n1,2\n"
        file_item = add_project_file(
            self.project["id"],
            name="sample.csv",
            content_base64=base64.b64encode(raw).decode("ascii"),
            mime_type="text/csv",
            size=len(raw),
            text_content=raw.decode("utf-8"),
            parent_folder_id=folder["id"],
        )
        stored = get_project_file(self.project["id"], file_item["id"])
        self.assertEqual(base64.b64decode(stored["contentBase64"]), raw)
        self.assertEqual(stored["textContent"], raw.decode("utf-8"))

        trash_items(
            self.project["id"],
            [{"type": "file", "id": file_item["id"]}],
        )
        detail = workspace_detail(self.project["id"])
        self.assertEqual(detail["files"], [])
        self.assertEqual(len(detail["trash"]), 1)

        restore_items(
            self.project["id"],
            [detail["trash"][0]["trashId"]],
        )
        detail = workspace_detail(self.project["id"])
        self.assertEqual(len(detail["files"]), 1)
        self.assertEqual(detail["trash"], [])

    def test_trashing_legacy_analysis_removes_it_from_core_project(self):
        cluster = store.add_cluster(
            self.project["id"],
            name="data",
            filename="data.csv",
            csv_text="x,y\n1,2\n",
        )
        analysis = store.save_analysis(
            self.project["id"],
            question_id="q",
            cluster_ids=[cluster["id"]],
            module_version="test",
            outcome={
                "status": "ok",
                "proposal": {"type": "next_observation", "input": {"x": 2}},
            },
        )
        self.assertEqual(len(store.project_detail(self.project["id"])["analyses"]), 1)
        trash_items(
            self.project["id"],
            [{"type": "analysis", "id": analysis["id"]}],
        )
        core = store.project_detail(self.project["id"])
        self.assertEqual(core["analyses"], [])
        self.assertEqual(core["proposals"], [])
        trash_types = {item["type"] for item in workspace_detail(self.project["id"])["trash"]}
        self.assertEqual(trash_types, {"analysis", "proposal"})


if __name__ == "__main__":
    unittest.main()
