import papermill as pm
import pytest

def test_custom_network_notebook_execution(tmp_path):
    """
    Tests the execution of the custom_network.ipynb notebook.
    """
    input_notebook = "examples/custom_network.ipynb"
    output_notebook = tmp_path / "output_custom_network.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed for {input_notebook}: {e}")
