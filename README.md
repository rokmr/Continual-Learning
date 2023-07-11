# Continual-Learning

Continual Learning is a concept to learn a model for a large number of tasks sequentially without forgetting knowledge obtained from the preceding tasks, where the data in the old tasks are not available anymore during training new ones. [Comprehensive Survey](https://arxiv.org/pdf/2302.00487.pdf)

**This repo is for Prompt-Based Continual Learning**

* [Learning to Prompt for Continual Learning](https://arxiv.org/pdf/2112.08654.pdf) || [GitHub](https://github.com/google-research/l2p) || [PyTorch] (https://github.com/JH-LEE-KR/l2p-pytorch)

In this method, learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions.
The effectiveness of L2P on multiple continual learning benchmarks, including class- and domain-incremental, and task-agnostic settings. 
The proposed L2P outperforms previous state-of-the-art methods consistently on all benchmarks.

* [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/pdf/2204.04799.pdf) || [GitHub](https://github.com/google-research/l2p) || [PyTorch](https://github.com/JH-LEE-KR/dualprompt-pytorch)

DualPrompt, a rehearsal-free continual learning approach to explicitly learn two sets of disjoint prompt spaces, G(eneral)-Prompt
and E(xpert)-Prompt, that encode task-invariant and task-specific instructions, respectively.
DualPrompt consistently sets state-of-the-art performance under the challenging class-incremental setting.

* [DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion
](https://arxiv.org/pdf/2111.11326.pdf) || [GitHub](https://github.com/arthurdouillard/dytox)

In this paper, we propose a transformer architecture based on a dedicated encoder/decoder framework. Critically, the
encoder and decoder are shared among all tasks. Through a dynamic expansion of special tokens, we specialize each
forward of our decoder network on a task distribution. Our strategy scales to a large number of tasks while having
negligible memory and time overheads due to strict control of the expansion of the parameters.





