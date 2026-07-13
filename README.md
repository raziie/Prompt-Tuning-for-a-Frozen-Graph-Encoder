# Prompt Tuning for a Frozen Graph Encoder

A parameter-efficient approach for adapting a self-supervised graph encoder to low-label node classification.

This repository contains the implementation and experiments from my Kaggle notebook, **Prompt Tuning for a Frozen Graph Encoder**, developed as an Advanced Deep Learning course project at Amirkabir University of Technology.

[View the original notebook on Kaggle](https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder)

---

## Overview

Graph neural networks can learn useful representations from graph-structured data, but fully fine-tuning a pretrained encoder may overfit when only a small number of labeled examples are available.

This project investigates whether a frozen, self-supervised graph encoder can instead be adapted using a lightweight trainable prompt.

The proposed pipeline consists of two stages:

1. A Graph Convolutional Network is pretrained with **Deep Graph Infomax**, using graph structure and node features without labels.
2. The pretrained encoder is frozen, and only an **additive feature prompt** and a **linear classifier** are optimized for downstream node classification.

This design preserves the pretrained graph representations while substantially reducing the number of trainable parameters.

---

## Method

### Self-supervised pretraining

A two-layer Graph Convolutional Network is pretrained using **Deep Graph Infomax (DGI)**.

DGI learns node representations by maximizing agreement between local node embeddings and a global graph-summary representation. No class labels are used during this stage.

### Frozen graph encoder

After pretraining, all GCN encoder parameters are frozen.

The frozen encoder maps node features to 128-dimensional embeddings:

```text
Input features → GCN layer (256) → ReLU → GCN layer (128)
```

Freezing the encoder reduces the risk of overfitting in the extreme low-label setting.

### Additive feature prompt

A shared trainable prompt vector is added to every node feature vector:

```text
X_prompt = X + p
```

The prompt acts as a global feature-space transformation that adapts the input to the frozen encoder without changing the encoder parameters.

### Downstream classifier

A linear classification head maps the frozen encoder embeddings to class logits.

During downstream training, gradients update only:

- The additive prompt vector
- The linear classifier

---

## Experimental setup

Experiments are conducted on three citation-network benchmarks:

| Dataset | Nodes | Input features | Classes | Labeled training nodes |
|---|---:|---:|---:|---:|
| Cora | 2,708 | 1,433 | 7 | 35 |
| CiteSeer | 3,327 | 3,703 | 6 | 30 |
| PubMed | 19,717 | 500 | 3 | 15 |

Only **five labeled nodes per class** are used for downstream training.

### Model configuration

| Component | Configuration |
|---|---|
| Encoder | Two-layer GCN |
| Hidden dimension | 256 |
| Embedding dimension | 128 |
| Pretraining objective | Deep Graph Infomax |
| Pretraining epochs | 200 |
| Prompt type | Additive feature prompt |
| Classifier | Linear layer |
| Optimizer | Adam |
| Pretraining learning rate | `1e-3` |
| Prompt-tuning learning rate | `1e-2` |
| Weight decay | `5e-4` |
| Evaluation | Three random seeds |

---

## Results

Prompt tuning improves test accuracy over the frozen-encoder baseline on all three datasets.

| Dataset | Frozen baseline | Additive prompt | Improvement |
|---|---:|---:|---:|
| Cora | 69.51% ± 3.16% | **76.55% ± 1.58%** | **+7.05 percentage points** |
| CiteSeer | 55.90% ± 4.02% | **58.34% ± 2.66%** | **+2.44 percentage points** |
| PubMed | 58.74% ± 8.75% | **61.89% ± 2.35%** | **+3.15 percentage points** |

In addition to improving average accuracy, prompt tuning reduces performance variance across the evaluated datasets.

### Parameter efficiency on Cora

For Cora, downstream adaptation trains only:

| Trainable component | Parameters |
|---|---:|
| Additive feature prompt | 1,433 |
| Linear classifier | 903 |
| **Total** | **2,336** |

The frozen graph encoder contains approximately 400,000 parameters, meaning that downstream adaptation updates less than one percent of the encoder size.

---

## Ablation study

The following Cora results examine the importance of pretraining, encoder freezing, and prompt design.

| Method | Test accuracy |
|---|---:|
| Random encoder + prompt | 59.08% ± 1.05% |
| Virtual-node prompt | 70.70% ± 3.66% |
| Full fine-tuning | 75.19% ± 1.04% |
| **Additive prompt with frozen DGI encoder** | **76.55% ± 1.58%** |

The results indicate that:

- Self-supervised pretraining is essential for effective prompt tuning.
- The additive feature prompt performs better than the virtual-node prompt.
- Full fine-tuning does not outperform the proposed method in the extreme low-label setting.
- Freezing the encoder may act as a useful form of regularization when labels are scarce.

---

## Effect of encoder architecture

Prompt-tuning performance also depends strongly on the quality of the pretrained encoder.

| Encoder | Test accuracy |
|---|---:|
| **GCN with DGI pretraining** | **76.55% ± 1.58%** |
| GraphSAGE with DGI pretraining | 68.92% ± 3.72% |
| GIN with DGI pretraining | 53.75% ± 3.57% |

These results suggest that prompt tuning cannot compensate for poor representation learning during pretraining. A well-structured pretrained embedding space remains important.

---

## Key findings

- A frozen self-supervised graph encoder can be adapted effectively using a small additive prompt.
- Prompt tuning consistently outperforms the frozen-encoder baseline across Cora, CiteSeer, and PubMed.
- On Cora, accuracy improves from **69.51% to 76.55%**.
- Only **2,336 parameters** are trained during downstream adaptation on Cora.
- The proposed approach slightly outperforms full fine-tuning under severe label scarcity.
- Performance depends strongly on the quality and architecture of the pretrained encoder.

---

## Running the notebook on Kaggle

1. Open the [Kaggle notebook](https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder).
2. Select **Copy & Edit**.
3. Configure the required accelerator if necessary.
4. Select **Run All** or execute the cells individually.

The required citation-network datasets can be downloaded automatically through the graph-learning libraries used in the notebook.

---

## Running locally

Clone your repository:

```bash
git clone https://github.com/raziie/Prompt-Tuning-for-a-Frozen-Graph-Encoder.git
cd prompt-tuning-graph-encoder
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the project notebook and run the cells in order.

> Package versions, hardware configuration, random seeds, and dataset versions may affect the reproduced results.

---

## Limitations

This study focuses on three citation-network datasets, which represent a relatively narrow class of graph-learning problems.

Additional limitations include:

- The use of only five labeled nodes per class introduces substantial evaluation variability.
- The shared additive prompt cannot represent node-specific or class-specific adaptations.
- Results may not directly generalize to larger, heterogeneous, dynamic, or domain-specific graphs.
- Prompt-tuning effectiveness depends on the quality of the pretrained encoder.

---

## Future work

Potential extensions include:

- Evaluating the method on larger and more diverse graph datasets
- Developing node-dependent or class-dependent prompts
- Exploring attention-based prompt mechanisms
- Testing alternative self-supervised graph-learning objectives
- Investigating why some encoder architectures transfer poorly when frozen
- Comparing parameter efficiency, memory use, and training time against full fine-tuning
- Studying robustness across a larger number of random data splits and seeds

---

## Author

**Razie Poorgholamrezaee**

Department of Computer Engineering  
Amirkabir University of Technology, Tehran, Iran

- Kaggle: [razie81](https://www.kaggle.com/razie81)
- Notebook: [Prompt Tuning for a Frozen Graph Encoder](https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder)

---

## Citation

When using or referring to this implementation, please cite the repository or the original Kaggle notebook:

```bibtex
@misc{poorgholamrezaee2026prompt,
  author       = {Razie Poorgholamrezaee},
  title        = {Prompt Tuning for a Frozen Graph Encoder},
  year         = {2026},
  howpublished = {Kaggle Notebook},
  url          = {https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder}
}
```

---

## License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided that the original copyright notice and license text are included.

See the LICENSE file for the full license terms.

Copyright © 2026 Razie.


