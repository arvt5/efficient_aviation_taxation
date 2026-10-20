# Environments

Archived cluster guide: Python 3.9.21, NumPy 2.0.2, PyTorch 2.8.0+cu128.
This is a historical guide record; the complete run-specific package freeze
has not yet been recovered.

Migration checks: Python 3.12.0, NumPy 2.4.6, Matplotlib 3.10.9, PyTorch 2.12.0
on macOS. No complete training run was performed with this environment.

The reproduction requirements pin the tested plotting/accounting environment.
PyTorch is optional for archived-output reproduction and required for solver
tests, training, and direct recertification. CUDA installations should follow
the chosen PyTorch release's platform-specific installation instructions.

PDF/PNG conversion of price-path SVGs uses librsvg's `rsvg-convert` executable.
