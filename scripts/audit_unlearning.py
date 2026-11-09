"""Compatibility entry point for the current EOS-corrected mask audits.

Legacy pre-fix results remain in saves/audits/implementation. New reports are
written separately to saves/audits/revision by scripts/audit_masks.py.
"""
from audit_masks import main

if __name__ == "__main__":
    main()
