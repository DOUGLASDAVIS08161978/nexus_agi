from setuptools import setup, Extension

ext = Extension(
    "miner_core",
    sources=["miner_core.c"],
    libraries=["crypto"],
    extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
)

setup(name="miner_core", version="1.0", ext_modules=[ext])
