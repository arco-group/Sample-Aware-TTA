# LDCT Mayo Clinic Example Checkpoints

This repository includes the exact checkpoints used by the LDCT Mayo Clinic example.

They are stored locally in `Sample-Aware-TTA/examples/ckp/`:

```text
  task_model/
    100_net_G_A.pth
  ae/
    epoch49/
      AE_input.pt
      AE_first_conv.pt
      AE_second_conv.pt
      AE_third_conv.pt
      AE_resnet_block_1.pt
      AE_resnet_block_2.pt
      AE_resnet_block_3.pt
      AE_resnet_block_4.pt
      AE_final_output.pt
```

The example command in the repository root README loads these files directly.
