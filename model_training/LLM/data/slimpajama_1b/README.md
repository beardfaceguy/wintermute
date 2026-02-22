---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: meta
    struct:
    - name: redpajama_set_name
      dtype: string
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 4066080183.08
    num_examples: 933130
  - name: validation
    num_bytes: 39109042
    num_examples: 9347
  - name: test
    num_bytes: 40114950
    num_examples: 9346
  download_size: 2424961933
  dataset_size: 4145304175.08
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
Sampled version of train split in [DKYoon/SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) resulting in about 1/6 of the pre-sampling train split. Validation and test splits remain the same.