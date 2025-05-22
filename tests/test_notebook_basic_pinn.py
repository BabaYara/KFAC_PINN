import papermill as pm
import pytest

def test_basic_pinn_notebook_execution(tmp_path):
    """
    Tests the execution of the basic_pinn.ipynb notebook.
    """
    input_notebook = "examples/basic_pinn.ipynb"
    output_notebook = tmp_path / "output_basic_pinn.ipynb"

    try:
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            kernel_name="python3"  # Specify the kernel if necessary
        )
    except Exception as e:
        pytest.fail(f"Notebook execution failed: {e}")

# To run this test, you would typically use pytest from your terminal:
# pytest tests/test_notebook_basic_pinn.py
