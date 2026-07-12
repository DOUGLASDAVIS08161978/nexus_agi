from setuptools import setup, Extension

ext = Extension(
    "miner_core",
    sources=["miner_core.c"],
    extra_compile_args=[
        "-O3",
        "-march=native",        # auto-enable SHA/NEON on ARM64
        "-funroll-loops",
        "-fomit-frame-pointer", # free an extra register
        "-ffast-math",          # no floats in SHA-256, safe and faster
    ],
)

setup(name="miner_core", version="2.0", ext_modules=[ext])
