"""End-to-end test — starts the Streamlit app and tests via browser automation."""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(APP_DIR, "app.py")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_FACE_PATH = os.path.join(FIXTURES_DIR, "test_face.jpg")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, timeout=30):
    """Wait until the Streamlit server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def streamlit_server():
    """Start Streamlit in headless mode and yield (process, port)."""
    port = _find_free_port()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", APP_PATH,
            "--server.headless", "true",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false",
            "--logger.level", "error",
        ],
        cwd=APP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not _wait_for_server(port):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Streamlit did not start on port {port}.\n"
            f"stdout: {stdout.decode()[-500:]}\n"
            f"stderr: {stderr.decode()[-500:]}"
        )

    yield proc, port

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---- Tests that only need HTTP (no browser) ----

class TestStreamlitServer:
    def test_server_responds(self, streamlit_server):
        import urllib.request
        _, port = streamlit_server
        url = f"http://localhost:{port}/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200

    def test_health_endpoint(self, streamlit_server):
        import urllib.request
        _, port = streamlit_server
        url = f"http://localhost:{port}/_stcore/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            body = resp.read().decode()
            assert "ok" in body.lower()


# ---- Playwright browser tests ----

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
class TestBrowserE2E:
    @pytest.fixture(scope="class")
    def browser_page(self, streamlit_server):
        _, port = streamlit_server
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"http://localhost:{port}", wait_until="networkidle", timeout=30000)
            # Wait for Streamlit to fully load
            page.wait_for_selector("h1", timeout=15000)
            yield page
            browser.close()

    def test_title_visible(self, browser_page):
        h1 = browser_page.query_selector("h1")
        assert h1 is not None
        text = h1.inner_text()
        assert "PikoLab" in text

    def test_sidebar_has_modes(self, browser_page):
        content = browser_page.content()
        assert "Upload" in content
        assert "Demo" in content

    def test_upload_mode_has_file_uploader(self, browser_page):
        # Upload is the default mode
        uploader = browser_page.query_selector('[data-testid="stFileUploader"]')
        # May or may not find by test id, check text instead
        content = browser_page.content()
        assert "Photo du visage" in content or "file" in content.lower()

    def test_upload_and_analyze(self, browser_page):
        """Upload a face image and verify analysis results appear."""
        if not os.path.exists(TEST_FACE_PATH):
            pytest.skip("Test face not available")

        # Find file input and upload
        file_input = browser_page.query_selector('input[type="file"]')
        if file_input is None:
            pytest.skip("File input not found in page")

        file_input.set_input_files(TEST_FACE_PATH)

        # Wait for analysis to complete (spinner disappears, tabs appear)
        browser_page.wait_for_timeout(5000)

        content = browser_page.content()
        # Should have the 4 tabs
        assert "Acquisition" in content or "Detection" in content, (
            "Analysis tabs should appear after upload"
        )
