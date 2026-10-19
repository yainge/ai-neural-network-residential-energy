"""Export model.pt to one self-contained model.onnx file.

Do not change the ONNX input/output names, shapes, dtype, or external_data flag.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model import ConsumptionPredictor

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "model.pt"
ONNX_OUT = HERE / "model.onnx"


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError("model.pt not found. Run: python starter/train.py")

    # Remove stale external-data files from older exports, if present.
    for stale in HERE.glob("model.onnx.data*"):
        stale.unlink()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = ConsumptionPredictor(mu=checkpoint["mu"], sigma=checkpoint["sigma"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 168, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT.as_posix(),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        external_data=False,
        dynamo=False,
    )

    external_files = list(HERE.glob("model.onnx.data*"))
    if external_files:
        raise RuntimeError(
            "The export created external tensor files. The platform accepts only one "
            "self-contained model.onnx file."
        )

    print(f"✓ ONNX exported: {ONNX_OUT}")
    print(f"  Size: {ONNX_OUT.stat().st_size / 1024:.1f} KB")
    print("Next step: python starter/validate_onnx.py")


if __name__ == "__main__":
    main()
