"""Select the native or explicitly requested reference-only distribution."""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


reference_only = os.environ.get("MAMUT_REFERENCE_ONLY", "0")
if reference_only not in {"0", "1"}:
    raise ValueError("MAMUT_REFERENCE_ONLY must be 0 (native) or 1 (Python only)")

extensions = []
if reference_only == "0":
    extensions = [
        Pybind11Extension(
            "mamut_routing_lib._static_exact",
            ["native/static/bindings.cpp"],
            cxx_std=20,
        ),
        Pybind11Extension(
            "mamut_routing_lib._td_exact",
            ["native/td/route.cpp"],
            include_dirs=["src/mamut_routing_lib/include"],
            depends=["src/mamut_routing_lib/include/mamut_routing_lib/td_exact.hpp"],
            libraries=["gmpxx", "gmp"],
            cxx_std=17,
        ),
    ]

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
