"""Allow running `python3 -m poisson_runner`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
