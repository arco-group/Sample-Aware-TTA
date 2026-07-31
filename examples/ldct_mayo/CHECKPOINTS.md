# LDCT Mayo Clinic Example Checkpoints

This repository includes the exact checkpoints used by the LDCT Mayo Clinic example.

They are stored locally in `weights/ckp/ldct_mayo/`:

```text
weights/ckp/ldct_mayo/
  task_model/
    100_net_G_A.pth
  ae/
    epoch49/
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

The example command in the repository root README loads these files directly.
