from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.runtime.coordinator import Coordinator, RemoteLease, server_for


class CoordinatorTest(unittest.TestCase):
    def test_expired_claim_is_orphaned_and_never_reassigned_or_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            coordinator = Coordinator(Path(temp) / "leases.sqlite")
            try:
                with patch("worldline.runtime.coordinator.time.time", return_value=100):
                    self.assertTrue(
                        coordinator.call("claim", {"job": "x", "signature": "v1", "owner": "first", "lease_sec": 1})[
                            "granted"
                        ]
                    )
                with patch("worldline.runtime.coordinator.time.time", return_value=102):
                    result = coordinator.call("claim", {"job": "x", "signature": "v1", "owner": "second"})
                    self.assertFalse(result["granted"])
                    self.assertEqual(result["status"], "orphaned")
                    with self.assertRaises(ValueError):
                        coordinator.call("heartbeat", {"job": "x", "owner": "first"})
                    coordinator.call("complete", {"job": "x", "owner": "first", "result": {"run_id": "original"}})
                    self.assertEqual(coordinator.call("status", {"job": "x"})["result"]["run_id"], "original")
                with self.assertRaisesRegex(ValueError, "signature"):
                    coordinator.call("claim", {"job": "x", "signature": "v2", "owner": "third"})
            finally:
                coordinator.db.close()

    def test_two_http_clients_have_one_owner_and_authenticated_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            coordinator = Coordinator(Path(temp) / "leases.sqlite")
            server = server_for(coordinator, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}"
            first, second = RemoteLease(endpoint, "job", "signature"), RemoteLease(endpoint, "job", "signature")
            try:
                self.assertTrue(first.claim()["granted"])
                self.assertFalse(second.claim()["granted"])
                first.complete({"run_id": "trial", "result_sha256": "hash"})
                self.assertEqual(second.call("status")["status"], "completed")
            finally:
                first.close()
                second.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                coordinator.db.close()
