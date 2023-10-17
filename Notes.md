## Folder Structure

```
Face-Transformer
├── Data
|   ├── casia-webface
|   |   ├── property
|   |   ├── train.idx
|   |   ├── train.lst
|   |   └── train.rec
|   └── ms1m-retinaface-t1
|       ├── property
|       ├── train.idx
|       ├── train.lst
|       └── train.rec
├── eval
|   ├── agedb_30.bin
|   ├── calfw.bin
|   ├── cfp_ff.bin
|   ├── cfp_fp.bin
|   ├── cplfw.bin
|   ├── lfw.bin
|   ├── sllfw.bin
|   └── talfw.bin
├── images
|   ├── arch.jpg
|   └── loss_function.png
├── util
|   ├── __init__.py
|   ├── test.py
|   ├── utils.py
|   └── verification.py
├── vit_pytorch
|   ├── __init__.py
|   ├── nat.py
|   ├── vit.py
|   ├── vit_face.py
|   └── vits_face.py
├── .gitignore
├── colab.ipynb
├── Face-Transformer.pdf
├── LICENSE
├── Notes.md
├── README.md
├── requirements.txt
├── run.ipynb
├── test_forward.py
├── test.py
└── train.py
```

## Meaning of flags

|  Flag   |         Argument         |                            Meaning                             |  Default   | Type  |
| :-----: | :----------------------: | :------------------------------------------------------------: | :--------: | :---: |
|  `-u`   | `--routing_table_status` |                      Routing Table Status                      |            |  str  |
|  `-w`   |      `--workers_id`      |                         GPU IDs or CPU                         |   `cpu`    |  str  |
|  `-e`   |        `--epochs`        |                        Training Epochs                         |    `1`     |  int  |
|  `-b`   |      `--batch_size`      |                           Batch Size                           |   `256`    |  int  |
|  `-d`   |      `--data_mode`       | Use which Database [`casia`, `vgg`, `ms1m`, `retina`, `ms1mr`] |   `ms1m`   |  str  |
|  `-n`   |         `--net`          |    Which Network [`VIT`, `VITs`, `SWT`, `NAT`, `EffNetV2m`]    |   `VITs`   |  str  |
| `-head` |         `--head`         |    Head Type [`Softmax`, `ArcFace`, `CosFace`, `SFaceLoss`]    | `ArcFace`  |  str  |
|  `-t`   |        `--target`        |                      Verification Targets                      |   `lfw`    |  str  |
|  `-r`   |        `--resume`        |                          Resume Model                          |            |  str  |
|         |        `--outdir`        |                        Output Directory                        |            |  str  |
|         |        `--model`         |                       Model Name for NAT                       | `nat_mini` |  str  |
|         |         `--opt`          |                           Optimizer                            |  `adamw`   |  str  |
|         |       `--opt-eps`        |                       Optimizer Epsilon                        |   `1e-8`   | float |
|         |      `--opt-betas`       |                        Optimizer Betas                         |            | float |
|         |       `--momentum`       |           Stochastic Gradient Descent (SGD) momentum           |   `0.9`    | float |
|         |     `--weight-decay`     |                          Weight Decay                          |   `0.05`   | float |

## Learning rate schedule parameters

|      Argument       |                    Meaning                     | Default  | Type  |
| :-----------------: | :--------------------------------------------: | :------: | :---: |
|      `--sched`      |                   Scheduler                    | `cosine` |  str  |
|       `--lr`        |                 Learning Rate                  |  `5e-4`  | float |
|    `--lr-noise`     |  Learning Rate Noise On/Off Epoch Percentages  |          | float |
|  `--lr-noise-pct`   |       Learning Rate Noise Limit Percent        |  `0.67`  | float |
|  `--lr-noise-std`   |     Learning Rate Noise Standard Deviation     |  `1.0`   | float |
|    `--warmup-lr`    |              Warmup Learning Rate              |  `1e-6`  | float |
|     `--min-lr`      | Lower Learning Rate Bound for Cyclic Scheduler |  `1e-5`  | float |
|  `--decay-epochs`   |     Epoch Interval to Decay Learning Rate      |   `30`   | float |
|  `--warmup-epochs`  |      Epoch to Warmup Decay Learning Rate       |   `3`    |  int  |
| `--cooldown-epochs` |   Epoch to Cooldown Learning Rate at min_lr    |   `10`   |  int  |
| `--patience-epochs` |    Patience Epochs for Plateau LR scheduler    |   `10`   |  int  |
|   `--decay-rate`    |              Decay Learning Rate               |  `0.1`   | float |

## Loss Function - CosFace (Softmax based loss function)

![Loss Function](./images/loss_function.png)
