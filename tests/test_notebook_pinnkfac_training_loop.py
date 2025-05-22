import papermill as pm
import pytest

def test_pinnkfac_training_loop_notebook_execution(tmp_path):
    """
    Tests the execution of the pinnkfac_training_loop.ipynb notebook.
    """
    input_notebook = "examples/pinnkfac_training_loop.ipynb"
    output_notebook = tmp_path / "output_pinnkfac_training_loop.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed for {input_notebook}: {e}")
