# Weights

Place task-model and reconstruction-model checkpoints here if you want to keep them next to the repo.

Example layout used by this repository:

```text
weights/
  ckp/
    ldct_mayo/
      task_model/
        100_net_G_A.pth
      ae/
        epoch49/
          AE_input_49.pt
          AE_first_conv_49.pt
          ...
```

The core code does not assume a fixed dataset or model name. The expected checkpoint paths are passed explicitly through the launcher or CLI flags.
