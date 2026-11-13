from __future__ import annotations

import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worldline.agents.transport import http_completion
from worldline.errors import ModelAPIError


class ModelTransportTest(unittest.TestCase):
    def server(self, delay, body):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                time.sleep(delay)
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return urllib.request.Request(f"http://127.0.0.1:{server.server_port}", data=b"{}")

    def test_slow_body_is_bounded_by_absolute_deadline(self):
        request = self.server(1, b"{}")
        start = time.monotonic()
        with self.assertRaisesRegex(ModelAPIError, "deadline"):
            http_completion(request, start + 0.2, 1024)
        self.assertLess(time.monotonic() - start, 0.8)

    def test_oversized_response_is_rejected_before_json_parse(self):
        request = self.server(0, b"x" * 10000)
        with self.assertRaisesRegex(ModelAPIError, "byte budget"):
            http_completion(request, time.monotonic() + 3, 1024)

    def test_controller_cancellation_interrupts_a_blocked_http_request(self):
        request = self.server(1, b"{}")
        event = threading.Event()
        timer = threading.Timer(0.1, event.set)
        timer.start()
        try:
            with self.assertRaisesRegex(ModelAPIError, "cancelled"):
                http_completion(request, time.monotonic() + 5, 1024, cancel_check=event.is_set)
        finally:
            timer.cancel()

    def test_valid_response_roundtrips_through_owned_io_process(self):
        request = self.server(0, b'{"choices":[]}')
        self.assertEqual(http_completion(request, time.monotonic() + 3, 1024), {"choices": []})
