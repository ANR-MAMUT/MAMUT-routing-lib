"""Include the manylinux wheel's native dependency notices in its PEP 639 metadata."""

from pathlib import Path
from shutil import copyfile
from subprocess import check_output
import sys


def main(project: Path) -> None:
    # cibuildwheel builds a disposable extraction of the source distribution.
    notices = project / "licenses" / "third-party"
    notices.mkdir(parents=True, exist_ok=True)
    for name in ("COPYING", "COPYING.LESSERv3", "COPYINGv2", "COPYINGv3"):
        copyfile(Path("/usr/share/licenses/gmp") / name, notices / f"GMP-{name}")
    copyfile("/usr/share/licenses/boost1.78/LICENSE_1_0.txt", notices / "Boost.txt")
    # Upstream pybind11 LICENSE, identical between v2.13.6 and v3.0.1.
    copyfile(Path(__file__).with_name("LICENSE.pybind11"), notices / "pybind11.txt")

    source_rpm = check_output(["rpm", "-q", "--qf", "%{SOURCERPM}", "gmp"], text=True)
    release = Path("/etc/almalinux-release").read_text().split()[2]
    (notices / "GMP-source.txt").write_text(
        "The wheel includes the GMP and GMP C++ shared libraries from AlmaLinux.\n"
        f"Corresponding source: https://vault.almalinux.org/{release}/BaseOS/Source/Packages/{source_rpm}\n",
        encoding="utf-8",
    )
    metadata = project / "pyproject.toml"
    text = metadata.read_text(encoding="utf-8")
    old = 'license = "MIT"\nlicense-files = ["LICENSE"]'
    new = (
        'license = "MIT AND BSL-1.0 AND BSD-3-Clause AND (LGPL-3.0-or-later OR GPL-2.0-or-later)"\n'
        'license-files = ["LICENSE", "licenses/third-party/*"]'
    )
    if text.count(old) != 1:
        raise ValueError("Source license metadata changed. Check the wheel's bundled licenses.")
    metadata.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
