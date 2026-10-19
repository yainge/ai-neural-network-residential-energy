"""Validate model.onnx before uploading it to the platform.

This intentionally mirrors the key server checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ONNX_PATH = HERE / "model.onnx"
MAX_BYTES = 25 * 1024 * 1024


def realistic_probe(batch: int) -> np.ndarray:
    x = np.zeros((batch, 168, 3), dtype=np.float32)
    hours = np.tile(np.arange(24, dtype=np.float32), 7)
    days = np.repeat(np.arange(7, dtype=np.float32), 24)
    for i in range(batch):
        x[i, :, 0] = 0.4 + 0.15 * np.sin(np.arange(168) * 2 * np.pi / 24 + i * 0.2)
        x[i, :, 1] = hours
        x[i, :, 2] = days
    return x


def reject_external_data(path: Path) -> None:
    model = onnx.load_model(path.as_posix(), load_external_data=False)
    tensors = list(model.graph.initializer)
    for sparse in model.graph.sparse_initializer:
        tensors.extend([sparse.values, sparse.indices])

    external = [
        tensor.name or "<unnamed>"
        for tensor in tensors
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
    ]
    if external:
        raise AssertionError(
            "External tensor data detected. Export with external_data=False. "
            f"Examples: {external[:5]}"
        )


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError("model.onnx not found. Run: python starter/export.py")
    if ONNX_PATH.stat().st_size > MAX_BYTES:
        raise AssertionError("model.onnx exceeds the platform 25 MB limit.")

    reject_external_data(ONNX_PATH)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    checks: list[tuple[str, bool, str]] = []

    def add(label: str, condition: bool, detail: str = "") -> None:
        checks.append((label, condition, detail))

    add("Exactly one input", len(inputs) == 1, f"found {len(inputs)}")
    add("Exactly one output", len(outputs) == 1, f"found {len(outputs)}")

    if len(inputs) == 1 and len(outputs) == 1:
        inp, out = inputs[0], outputs[0]
        add("Input name = input", inp.name == "input", f"got {inp.name}")
        add("Output name = output", out.name == "output", f"got {out.name}")
        add("Input dtype = float32", inp.type == "tensor(float)", f"got {inp.type}")
        add("Output dtype = float32", out.type == "tensor(float)", f"got {out.type}")
        add(
            "Input shape = [batch,168,3]",
            len(inp.shape) == 3 and inp.shape[1] == 168 and inp.shape[2] == 3,
            f"got {inp.shape}",
        )
        add(
            "Output shape = [batch,24]",
            len(out.shape) == 2 and out.shape[1] == 24,
            f"got {out.shape}",
        )

        for batch in (1, 4):
            try:
                y = session.run([out.name], {inp.name: realistic_probe(batch)})[0]
                add(f"Batch {batch} shape", y.shape == (batch, 24), f"got {y.shape}")
                add(f"Batch {batch} finite", np.isfinite(y).all(), "NaN/Inf detected")
                add(
                    f"Batch {batch} range < ±160 kWh",
                    float(np.abs(y).max(initial=0.0)) < 160.0,
                    f"max |y|={float(np.abs(y).max(initial=0.0)):.3f}",
                )
            except Exception as exc:
                add(f"Batch {batch} inference", False, str(exc))

    failures = 0
    print(f"Validating {ONNX_PATH.name}\n")
    for label, ok, detail in checks:
        if ok:
            print(f"  ✓ {label}")
        else:
            failures += 1
            suffix = f": {detail}" if detail else ""
            print(f"  ✗ {label}{suffix}")

    print()
    if failures:
        raise SystemExit(f"✗ {failures} validation check(s) failed.")

    print("✓ Model is compatible with the platform contract.")
    print("✓ Upload ONLY starter/model.onnx to the Residential Energy Challenge.")


if __name__ == "__main__":
    main()
