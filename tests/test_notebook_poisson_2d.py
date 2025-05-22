import papermill as pm
import pytest

def test_poisson_2d_notebook_execution(tmp_path):
    """
    Tests the execution of the poisson_2d.ipynb notebook.
    """
    input_notebook = "examples/poisson_2d.ipynb"
    output_notebook = tmp_path / "output_poisson_2d.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed for {input_notebook}: {e}")
