# Continual-Learning

Continual Learning is a concept to learn a model for a large number of tasks sequentially without forgetting knowledge obtained from the preceding tasks, where the data in the old tasks are not available anymore during training new ones.

**This repo is for Prompt-Based Continual Learning**

* [Learning to Prompt for Continual Learning](https://arxiv.org/pdf/2112.08654.pdf) || [GitHub](https://github.com/google-research/l2p)

In this method, learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions. In our proposed framework, prompts are small learnable parameters, which are maintained in a memory space. The objective is to optimize prompts to instruct the model prediction and explicitly manage task-invariant and task-specific knowledge while maintaining model plasticity.

The effectiveness of L2P on multiple continual learning benchmarks, including class- and domainincremental, and task-agnostic settings. 
The proposed L2P outperforms previous state-of-the-art methods consistently on all benchmarks.

* [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning] (https://arxiv.org/pdf/2204.04799.pdf) || [GitHub](https://github.com/google-research/l2p)

DualPrompt, a rehearsal-free continual learning approach to explicitly learn two sets of disjoint prompt spaces, G(eneral)-Prompt
and E(xpert)-Prompt, that encode task-invariant and task-specific instructions, respectively.
DualPrompt consistently sets state-of-the-art performance under the challenging class-incremental setting.



