# Radia

3D Magnetostatics Computer Code

## Installing Radia

Radia is a 3D magnetostatics code with a C++ core (`cpp/src/core`) and client
interfaces for **Python**, **Mathematica** (via MathLink), and **Igor Pro**.
This document covers building from source and using the prebuilt binaries
shipped in `env/`.

### Repository layout

| Path                | Contents                                                                      |
| ------------------- | ----------------------------------------------------------------------------- |
| `cpp/src/`          | C++ core, extensions (`triangle`, `auxparse`, `genmath`), and client bindings |
| `cpp/gcc/`          | GCC/Linux Makefile — builds the static library `libradia.a`                   |
| `cpp/py/`           | Python extension build (`setup.py` + Makefile)                                |
| `cpp/vc/`           | Visual Studio solution (`Radia.sln`) for Windows builds                       |
| `ext_lib/`          | Prebuilt third-party libraries (FFTW, MathLink, Igor XOP support, ...)        |
| `env/radia_python/` | Python examples, utilities, and prebuilt `radia` modules                      |
| `env/radia_math/`   | Radia for Mathematica package (`Radia/`, `Radia.zip`)                         |
| `env/radia_igor/`   | Radia for Igor Pro package (`Radia.xop`, procedures, help)                    |

---

### 1. Python module — build from source (Linux)

#### Prerequisites

- `gcc`/`g++` (or `cc`/`c++`) and GNU `make`
- `wget` and `tar` (used to fetch and unpack FFTW if needed)
- Python (3.x) with development headers and `setuptools`

The top-level `Makefile` orchestrates the whole build:

```sh
## Full build: download & compile FFTW 2.1.5 (single precision), then the
## C++ core, then the Python extension
make all

## If ext_lib/libfftw.a already exists, skip the FFTW step:
make            # equivalent to `make nofftw`

## Clean up intermediates and built libraries
make clean
```

What each stage does:

1. **`make fftw`** — downloads `fftw-2.1.5.tar.gz` (from the SRW repository),
   configures it with `--enable-float --with-pic`, and installs
   `ext_lib/libfftw.a`. Radia requires FFTW **2.1.5 compiled in single
   precision**; a modern system FFTW 3.x will not work.
2. **`make core`** — runs `cpp/gcc/Makefile` to compile the core into
   `cpp/gcc/libradia.a`.
3. **`make pylib`** — runs `python setup.py build_ext` in `cpp/py/`, links
   against `libradia.a` and `libfftw.a`, and copies the resulting module to
   `env/radia_python/radia.so`.

Note: `cpp/py/Makefile` invokes `python`; if your system only provides
`python3`, either adjust that Makefile or build in an environment (venv/conda)
where `python` is the interpreter you want the module built for. The module is
specific to the Python version it was built with.

#### Verify the build

```sh
cd env/radia_python
python RADIA_Example01.py
```

(The plotting utilities `uti_plot*.py` use `matplotlib` if available.)

#### Use it in your own code

Add `env/radia_python` to your `PYTHONPATH` (or copy `radia.so` next to your
scripts / into `site-packages`):

```sh
export PYTHONPATH=/path/to/Radia/env/radia_python:$PYTHONPATH
python -c "import radia; print(radia.UtiVer())"
```

#### Optional: MPI-enabled build

The core and Python extension can be built with MPI support for parallel
relaxation/field computation:

```sh
make MODE=mpi        # OpenMPI (include/lib paths as used on BNL/NSLS-II servers)
make MODE=mpi_nersc  # MPICH (uses $MPICH_DIR, as on NERSC clusters)
```

The MPI header/library paths are hardcoded for those environments — on other
systems, edit `MPI_INCLUDE_DIR_*` in `cpp/gcc/Makefile` and the
`mpi`/`mpi_nersc` branches in `cpp/py/setup.py` to match your MPI installation.
MPI can be initialized either by Radia itself or from `mpi4py` (see
`env/radia_python/RADIA_TestParallel01_mpi4py.py` and `RADIA_TestParallel01.py`).

#### Building with conda-forge dependencies

The compiler toolchain, Python, and MPI can all come from a conda-forge
environment, so nothing needs to be installed system-wide and no files need to
be downloaded at build time:

```sh
conda create -n radia -c conda-forge compilers make python setuptools
conda activate radia
make all
```

Notes:

- **FFTW is still built from the vendored tarball.** Radia requires the
  **FFTW 2 API in single precision** — `cpp/src/ext/genmath/gmfft.cpp` uses
  the FFTW 2-only plan interface (`fftw_create_plan`, `fftwnd`, ...), and the
  bundled header `cpp/src/ext/genmath/fftw.h` hardcodes `FFTW_ENABLE_FLOAT`,
  making `fftw_real` a `float`. FFTW 3 removed that API, so conda-forge's
  `fftw` package (3.x) is not a substitute; 2.1.5 is the last FFTW 2 release.
  The `make fftw` stage uses `ext_lib/fftw-2.1.5.tar.gz`, which is committed
  in this repository — the download in the Makefile only triggers if that
  tarball is missing, so the build works offline.
- Run `make` inside the activated environment so `python` resolves to the
  conda interpreter — the module is built for whatever `python` is on `PATH`.

For an MPI build, add MPI from conda-forge as well:

```sh
conda install -c conda-forge openmpi mpi4py   # or: mpich mpi4py
```

The `MODE=mpi` / `MODE=mpi_nersc` paths hardcoded in `cpp/gcc/Makefile` and
`cpp/py/setup.py` are for BNL/NERSC systems, so point them at the conda
environment instead:

1. In `cpp/gcc/Makefile`, set the MPI include path to the conda environment,
   e.g. replace the `MPI_INCLUDE_DIR_*` values with `$(CONDA_PREFIX)/include`.
2. In the `mpi` branch of `cpp/py/setup.py`, replace
   `/usr/lib/openmpi/include` and `/usr/lib/openmpi/lib` with
   `os.environ['CONDA_PREFIX'] + '/include'` and `... + '/lib'`, and adjust
   the `libraries` list to what your MPI provides: `mpi` for OpenMPI ≥ 5
   (which dropped the C++ bindings, so drop `mpi_cxx`), `mpi_cxx` + `mpi` for
   OpenMPI 4.x, or `mpicxx` + `mpi` for MPICH.
3. Build with `make MODE=mpi`.

At runtime, `LD_LIBRARY_PATH` is usually unnecessary inside an activated
conda environment, and the MPI examples run with the conda-provided `mpiexec`
(see `RADIA_TestParallel01_mpi4py.py` for starting MPI via `mpi4py`).

#### macOS note

The GCC Makefiles target Linux (`-DLINUX`, `-lrt`, `-luuid`, GNU `sed -i`
syntax in the FFTW step), so a source build on macOS requires minor Makefile
edits (drop `-lrt`/`-luuid`, use `sed -i ''`, remove/replace `-DLINUX`).
This applies to conda-forge builds on macOS too, where the compilers are
Clang-based.

---

### 2. Python module — prebuilt binaries

`env/radia_python/` ships prebuilt modules for several platforms and Python
versions, named `radia_py<pyver>_<arch>[_mpi].{so,pyd}`, e.g.:

- `radia_py3_12_x64.pyd` — Windows 64-bit, Python 3.12
- `radia_py3_8_x64.pyd` / `radia_py3_8_x64_mpi.pyd` — Windows 64-bit, Python 3.8
- `radia_py3_7_x86_64.so` / `radia_py3_7_x86_64_mpi.so` — Linux 64-bit, Python 3.7

To use one, copy it to `radia.pyd` (Windows) or `radia.so` (Linux) in a
directory on your `PYTHONPATH` — the version matching your interpreter:

```sh
cp radia_py3_7_x86_64.so radia.so     # Linux, Python 3.7
copy radia_py3_12_x64.pyd radia.pyd   # Windows, Python 3.12
```

---

### 3. Windows build (Visual Studio)

Open `cpp/vc/Radia.sln` in Visual Studio. The solution contains projects for
each target:

- `RadiaLib` — the static core library
- `RadiaPython` — the Python module (`radia.pyd`)
- `RadiaMath` — the Mathematica MathLink executable (`Radia.exe`)
- `RadiaIgor` — the Igor Pro XOP (`Radia.xop`)

Prebuilt Windows dependencies are in `ext_lib/` (`fftw64_f.lib`, GLUT,
libpng, zlib, MathLink under `ext_lib/mathlink/`, Igor XOP support under
`ext_lib/igor/`). Build the `x64 Release` configuration; for `RadiaPython`,
point the project's include/library paths at your Python installation.

---

### 4. Radia for Mathematica

Requires Mathematica 9.0+ on Windows, Linux, or macOS (64-bit).

1. Take the package from `env/radia_math/Radia/` (or unzip
   `env/radia_math/Radia.zip`).
2. Copy the whole `Radia` folder (with subfolders) into Mathematica's
   applications directory, e.g. `.../AddOns/Applications/Radia` of your
   Mathematica installation (or `$UserBaseDirectory/Applications/Radia`).
   The folder must contain `Radia.exe` (the MathLink binary), `init.m`,
   `RadiaTopics.nb`, and `RadiaButtons.nb`.
3. Load it in a notebook with ``<<Radia` `` and verify with the examples in
   `Radia/Examples/Example#1.nb` through `Example#7.nb`.

See `env/radia_math/Radia/ReadMe.txt` and the HTML documentation in
`Radia/RadiaDocHTML/` for details.

---

### 5. Radia for Igor Pro (Windows)

Requires Igor Pro (see `env/radia_igor/Radia/ReadMe.txt` for supported
versions). In short:

1. Remove any previous Radia files/shortcuts from the Igor Pro folder and its
   `Igor Extensions`, `Igor Help Files`, and `Igor Procedures` subfolders.
2. Quit Igor Pro and copy `env/radia_igor/Radia/` into the Igor Pro folder,
   then create shortcuts:
   - `Radia.xop` → `Igor Pro Folder/Igor Extensions`
   - `Radia Help.ihf` → `Igor Pro Folder/Igor Help Files`
   - `Radia Procedures` (e.g. `Radia Menu.ipf`) → `Igor Pro Folder/Igor Procedures`
3. Start Igor Pro; compile the Radia help file if prompted. A "Radia..." menu
   should appear. Examples are under `Radia Procedures/Examples/`.

---

### Documentation

- HTML reference guides (Mathematica and Igor flavors) are bundled under
  `env/radia_math/Radia/RadiaDocHTML/` and `env/radia_igor/Radia/RadiaDocHTML/`.
- Papers describing the methods are in `literature/`.
- Python API usage is best learned from `env/radia_python/RADIA_Example0*.py`.
