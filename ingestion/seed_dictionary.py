"""DEPRECATED — alarm_dictionary was removed in the PIADE-only pivot.

The project now uses PIADE sequences_1h_data.csv exclusively.
There is no alarm_dictionary table; context is derived from telemetry windows.
This file is kept as a tombstone and does nothing when executed.
"""

import sys


def main() -> None:
    print(
        "seed_dictionary is deprecated and has no effect. "
        "The alarm_dictionary table no longer exists in this project.",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
