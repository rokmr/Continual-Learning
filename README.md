# Continual-Learning

Continual Learning is a concept to learn a model for a large number of tasks sequentially without forgetting knowledge obtained from the preceding tasks, where the data in the old tasks are not available anymore during training new ones.

**This repo is for Prompt-Based Continual Learning**

[Learning to Prompt for Continual Learning](https://arxiv.org/pdf/2112.08654.pdf)
In this method, learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions. In our proposed framework, prompts are small learnable parameters, which are maintained in a memory space. The objective is to optimize prompts to instruct the model prediction and explicitly manage task-invariant and task-specific knowledge while maintaining model plasticity.
