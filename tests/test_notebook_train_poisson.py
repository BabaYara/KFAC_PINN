import papermill as pm
import pytest

def test_train_poisson_notebook_execution(tmp_path):
    """
    Tests the execution of the train_poisson.ipynb notebook.
    """
    input_notebook = "examples/train_poisson.ipynb"
    output_notebook = tmp_path / "output_train_poisson.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed for {input_notebook}: {e}")
