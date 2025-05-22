import papermill as pm
import pytest

def test_forward_laplacian_demo_notebook_execution(tmp_path):
    """
    Tests the execution of the forward_laplacian_demo.ipynb notebook.
    """
    input_notebook = "examples/forward_laplacian_demo.ipynb"
    output_notebook = tmp_path / "output_forward_laplacian_demo.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed for {input_notebook}: {e}")
