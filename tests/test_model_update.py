import subprocess
import pytest

def test_model_update_residual():
    try:
        subprocess.run(["python", "model_update.py", "--model", "residual", "--pred_only"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"model_update.py --model residual failed with error: {e.stderr.decode()}")