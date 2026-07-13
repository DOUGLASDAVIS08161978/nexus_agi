import platform
from setuptools import setup, Extension

machine = platform.machine()
if machine in ("aarch64", "arm64"):
    # Termux/Android clang ignores -march=native for crypto extensions;
    # explicit +sha2 is required to define __ARM_FEATURE_SHA2 and enable
    # the vsha256* hardware intrinsics on the phone's dedicated SHA silicon.
    extra_compile_args = [
        "-O3",
        "-march=armv8-a+sha2",
        "-mtune=native",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-fno-stack-protector",   # no canary overhead in hot loop
        "-fipa-cp",               # constant propagation across inlined calls
    ]
else:
    extra_compile_args = [
        "-O3",
        "-march=native",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-ffast-math",
    ]

ext = Extension("miner_core", sources=["miner_core.c"],
                extra_compile_args=extra_compile_args)

setup(name="miner_core", version="3.1", ext_modules=[ext])
