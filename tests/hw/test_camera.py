import shutil, subprocess, pytest
pytestmark = pytest.mark.hw

def test_rpicam_vid_exists():
    assert shutil.which("rpicam-vid") is not None

def test_rpicam_mjpeg_short_capture():
    proc = subprocess.run(
        ["rpicam-vid","-n","--codec","mjpeg","-t","1000","-o","-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
    )
    assert len(proc.stdout) > 0
