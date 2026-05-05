#!/usr/bin/env python3
"""Compatibility shim for the MCE-IRL primer generator.

Use ``mce_irl_run.py`` for the known-truth validation workflow.
"""

from __future__ import annotations

from mce_irl_run import main


if __name__ == "__main__":
    main()
