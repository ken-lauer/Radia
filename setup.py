"""
Pip-installable build for the Radia Python extension.

Runs the repository's regular `make` stages (FFTW if needed, then the C++
core into cpp/gcc/libradia.a) and then compiles/links the `radia` extension
itself, so the module always matches the interpreter pip is installing into.
Set MODE=mpi (or mpi_nersc) in the environment for the MPI variants, matching
the top-level Makefile.
"""

import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()
CPP = ROOT / "cpp"

ext_kwargs = {
    "define_macros": [("MAJOR_VERSION", "1"), ("MINOR_VERSION", "0")],
    "include_dirs": [
        str(CPP / "src" / "lib"),
        str(CPP / "src" / "ext" / "auxparse"),
        str(CPP / "src" / "core"),
    ],
    "libraries": ["radia", "m", "fftw"],
    "library_dirs": [str(CPP / "gcc"), str(ROOT / "ext_lib")],
    "sources": [str(CPP / "src" / "clients" / "python" / "radpy.cpp")],
}

MODE = os.environ.get("RADIA_MODE", "0")
if MODE == "mpi":
    mpi_root = os.environ.get("CONDA_PREFIX", "/usr/lib/openmpi")
    ext_kwargs["include_dirs"].append(os.path.join(mpi_root, "include"))
    ext_kwargs["libraries"] += ["mpi", "dl"]
    ext_kwargs["library_dirs"].append(os.path.join(mpi_root, "lib"))
elif MODE == "mpi_nersc":
    ext_kwargs["libraries"] += ["mpichcxx_intel", "dl"]
    ext_kwargs["library_dirs"].append(os.path.join(os.environ["MPICH_DIR"], "lib"))
elif MODE != "0":
    raise ValueError(f"Unknown Radia compilation/linking option: MODE={MODE!r}")


class RadiaBuildExt(build_ext):
    """
    Build the FFTW and Radia core static libraries via make, then build the
    Python extension normally.
    """

    def run(self):
        # Always recompile/relink: setuptools' timestamp check cannot see that
        # libradia.a or the RADIA_MODE flags changed since the last build.
        self.force = True
        env = dict(os.environ, MODE=MODE)
        if not (ROOT / "ext_lib" / "libfftw.a").exists():
            subprocess.run(["make", "fftw"], cwd=ROOT, check=True, env=env)
        subprocess.run(["make", "core"], cwd=ROOT, check=True, env=env)
        super().run()


setup(
    ext_modules=[Extension("radia", **ext_kwargs)],
    cmdclass={"build_ext": RadiaBuildExt},
)
