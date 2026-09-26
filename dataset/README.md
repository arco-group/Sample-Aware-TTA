# Dataset

Put your dataset here or point the launcher to an external location with `DATA_ROOT`.

The core TTA code does not assume a fixed dataset layout. The [example documentation](../examples/README.md) describes one concrete low-dose CT setup.

The bundled `paired_slice` loader accepts either `patient/modality/slice_N.npy` or
`patient/series/modality/slice_N.npy`. Slice filenames must use the same integer `N`
across modalities. Arrays must be `.npy` files normalized to `[0, 1]`; the loader maps
them to `[-1, 1]` before inference.

```text
dataset_root/
  patient_001/
    LDCT/
      slice_0.npy
    HDCT/
      slice_0.npy
```
