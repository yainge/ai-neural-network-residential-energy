"""Export model.pt to one self-contained model.onnx file.

Do not change the ONNX input/output names, shapes, dtype, or external_data flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lstm_models import MODEL_NAMES, load_checkpoint_model

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "model.pt"
ONNX_OUT = HERE / "model.onnx"


def main(
    model_name: str | None = None,
    checkpoint_path: Path = CHECKPOINT,
    onnx_out: Path = ONNX_OUT,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError("model.pt not found. Run: python starter/train.py")

    # Remove stale external-data files from older exports, if present.
    for stale in onnx_out.parent.glob(f"{onnx_out.name}.data*"):
        stale.unlink()

    model, _, model_name = load_checkpoint_model(checkpoint_path, "cpu", model_name)
    print(f"Model: {model_name}")

    dummy = torch.zeros(1, 168, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_out.as_posix(),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        external_data=False,
        dynamo=False,
    )

    external_files = list(onnx_out.parent.glob(f"{onnx_out.name}.data*"))
    if external_files:
        raise RuntimeError(
            "The export created external tensor files. The platform accepts only one "
            "self-contained model.onnx file."
        )

    print(f"✓ ONNX exported: {onnx_out}")
    print(f"  Size: {onnx_out.stat().st_size / 1024:.1f} KB")
    print("Next step: python starter/validate_onnx.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model.pt to model.onnx.")
    parser.add_argument(
        "--model", default=None, choices=MODEL_NAMES,
        help="Only needed for old checkpoints that do not store the model name.",
    )
    args = parser.parse_args()
    main(model_name=args.model)

