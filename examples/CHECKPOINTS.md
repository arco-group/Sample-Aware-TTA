# LDCT Mayo Clinic checkpoints

Checkpoint binaries are not included. Download or copy them into this canonical layout,
or pass equivalent external locations through `--task_checkpoint_path` and
`--ae_checkpoint_dir`:

```text
examples/ckp/
  task_model/100_net_G_A.pth
  ae/epoch49/
    AE_input_49.pt
    AE_first_conv_49.pt
    AE_second_conv_49.pt
    AE_third_conv_49.pt
    AE_resnet_block_1_49.pt
    AE_resnet_block_2_49.pt
    AE_resnet_block_3_49.pt
    AE_resnet_block_4_49.pt
    AE_final_output_49.pt
```

The loader constructs each AE filename as `AE_{name}_{ae_epoch}.pt`; therefore the
directory above must be passed with `--ae_epoch 49`.
