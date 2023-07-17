# Continual-Learning

Continual Learning is a concept to learn a model for a large number of tasks sequentially without forgetting knowledge obtained from the preceding tasks, where the data in the old tasks are not available anymore during training new ones. [Comprehensive Survey](https://arxiv.org/pdf/2302.00487.pdf)

## There are three common continual learning scenarios: 

1. **Task-incremental learning** : TIL is provided with task indexes for inference and thus using task-specific neural networks could overcome forgetting.

2. **Class-incremental learning** : CIL increases classes in sequence with task indexes being unknownfor inference. Nevertheless, the classes are generally from the same domain, which more or less decreases the challenge.

3. **Domain-incremental learning** : DIL, where classes keep the same but the involved domains commonly vary a lot in sequence, with task indexes being not provided for inference.

**This repo is for Prompt-Based Continual Learning**

* [Learning to Prompt for Continual Learning](https://arxiv.org/pdf/2112.08654.pdf) || [GitHub](https://github.com/google-research/l2p) || [PyTorch](https://github.com/JH-LEE-KR/l2p-pytorch)

In this method, learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions.
The effectiveness of L2P on multiple continual learning benchmarks, including class- and domain-incremental, and task-agnostic settings. 
The proposed L2P outperforms previous state-of-the-art methods consistently on all benchmarks.
![MODEL](https://github.com/google-research/l2p/blob/main/l2p_illustration.png)

* [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/pdf/2204.04799.pdf) || [GitHub](https://github.com/google-research/l2p) || [PyTorch](https://github.com/JH-LEE-KR/dualprompt-pytorch)

DualPrompt, a rehearsal-free continual learning approach to explicitly learn two sets of disjoint prompt spaces, G(eneral)-Prompt
and E(xpert)-Prompt, that encode task-invariant and task-specific instructions, respectively.
DualPrompt consistently sets state-of-the-art performance under the challenging class-incremental setting.
![MODEL](https://github.com/google-research/l2p/blob/main/dualprompt_illustration.png)

* [DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion
](https://arxiv.org/pdf/2111.11326.pdf) || [GitHub](https://github.com/arthurdouillard/dytox)

In this paper, we propose a transformer architecture based on a dedicated encoder/decoder framework. Critically, the
encoder and decoder are shared among all tasks. Through a dynamic expansion of special tokens, we specialize each
forward of our decoder network on a task distribution. Our strategy scales to a large number of tasks while having
negligible memory and time overheads due to strict control of the expansion of the parameters.
![Model](https://github.com/arthurdouillard/dytox/blob/main/images/dytox.png)

* [CODA-Prompt: COntinual Decomposed Attention-based Prompting for
Rehearsal-Free Continual Learning](https://arxiv.org/pdf/2211.13218.pdf) || [GitHub](https://github.com/GT-RIPL/CODA-Prompt/tree/main)

The author propose to learn a set of prompt components which are assembled with input-conditioned weights to produce input-conditioned prompts,
resulting in a novel attention-based end-to-end key-query scheme.
![Model](https://github.com/GT-RIPL/CODA-Prompt/blob/main/method_coda-p.png)

* [AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning](https://arxiv.org/pdf/2305.11488.pdf) || [GitHub](https://github.com/bhrqw/AttriCLIP)

AttriCLIP is built upon the pre-trained visual-language model CLIP.
AttriCLIP, which is a prompt tuning approach for continual learning based on CLIP. We train
different prompts according to the attributes of images to avoid knowledge overwriting caused by training the
same model in sequence of classes.

* [S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning](https://arxiv.org/pdf/2207.12819.pdf) || [GitHub](https://github.com/iamwangyabin/S-Prompts)

In this paper, we explore a rule-breaking idea to instead play a win-win game, i.e., learning the
prompts independently across domains so that the prompting can achieve the best for each domain.
This setup is for **DIL.**
![Model](https://github.com/iamwangyabin/S-Prompts/blob/main/SPrompts.png)





