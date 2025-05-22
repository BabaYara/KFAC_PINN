import jax
import jax.numpy as jnp

def f(x):
    return x * jnp.abs(x)

try:
    g = jax.grad(f)
    val_g = g(0.0)
    print(f"g(0.0) = {val_g}")

    gg = jax.grad(g)
    val_gg = gg(0.0)
    print(f"gg(0.0) = {val_gg}")
    print("Successfully computed first and second derivatives at 0.0.")
except Exception as e:
    print(f"An error occurred: {e}")
