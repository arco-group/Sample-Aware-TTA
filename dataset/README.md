# Dataset

Put your dataset here or point the launcher to an external location with `DATA_ROOT`.

The core TTA code does not assume a fixed dataset layout. The example launcher in `examples/ldct_mayo/` documents one concrete low-dose CT setup.

If you use the bundled slice-based loader, keep the expected `.npy` slice structure consistent with that loader's documentation.
