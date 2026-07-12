# Prompt Tuning for a Frozen Graph Encoder

This repository contains the code and experiments from my Kaggle notebook, **Prompt Tuning for a Frozen Graph Encoder**.

The project explores a parameter-efficient training approach in which the graph encoder remains frozen while task-specific prompt parameters are optimized. This allows the pretrained encoder to be adapted without updating all of its parameters.

## Project objectives

* Explore prompt tuning for graph representation learning.
* Keep the pretrained graph encoder frozen during training.
* Train a smaller set of task-specific parameters.
* Examine the effectiveness and efficiency of prompt-based adaptation.
* Provide a reproducible notebook for further experimentation.

## Notebook

The original notebook is available on Kaggle:

[Prompt Tuning for a Frozen Graph Encoder](https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder)

The notebook contains the complete workflow, including data preparation, model configuration, training, and evaluation.

## Running on Kaggle

To run the original Kaggle version:

1. Open the Kaggle notebook.
2. Select **Copy & Edit**.
3. Configure the required accelerator and data sources.
4. Select **Run All** or execute the cells individually.

## Author

**Razie_81**

* Kaggle: [razie81](https://www.kaggle.com/razie81)
* Original notebook: [Prompt Tuning for a Frozen Graph Encoder](https://www.kaggle.com/code/razie81/prompt-tuning-for-a-frozen-graph-encoder)

