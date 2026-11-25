"""Print the installed CMake configuration directory for MAMUTRouting."""

from pathlib import Path


def get_cmake_dir() -> Path:
    return Path(__file__).resolve().with_name("cmake")


if __name__ == "__main__":
    print(get_cmake_dir())
