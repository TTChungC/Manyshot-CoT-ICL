#!/usr/bin/env python3
"""MATH reinforce inference (first-answer demo pool)."""

import sys

from cot_icl.runners import math_reinforce

if __name__ == "__main__":
    if "--pool-variant" not in sys.argv:
        sys.argv[1:1] = ["--pool-variant", "firstans"]
    math_reinforce.main()
