"""Resume the LowConfidence queue with the literal v4 reference-log path."""

from pathlib import Path

import run_low_confidence as entry


def configure_with_v4_reference(protocol, output, reference_copy):
    key = protocol["retain_only_model"]["evaluation_logs"]
    original = (entry.FORMAL_OUTPUT / "frozen_inputs" / Path(key).name).resolve()
    expected = protocol["inputs"][key]
    entry.require(entry.sha256(reference_copy) == entry.sha256(original) == expected,
                  "Retain-only reference differs from the frozen v4 input")
    return ORIGINAL_CONFIGURE(protocol, output, original)


ORIGINAL_CONFIGURE = entry.configure_eval

if __name__ == "__main__":
    entry.configure_eval = configure_with_v4_reference
    entry.__file__ = str(Path(__file__).resolve())
    entry.main()
