#!/usr/bin/env python3
"""GSM8K reinforce inference (wrong-answer demo pool)."""

import sys

from cot_icl.runners import gsm8k_reinforce

if __name__ == "__main__":
    if "--pool-variant" not in sys.argv:
        sys.argv[1:1] = ["--pool-variant", "wrongans"]
    gsm8k_reinforce.main()
