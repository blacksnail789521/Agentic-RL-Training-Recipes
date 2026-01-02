# Agentic-RL-Training-Recipes
Training Recipes for Agentic Reinforcement Learning in LLMs: A Survey

[![arXiv](https://img.shields.io/badge/arXiv-pending-lightgrey.svg)](#)
[![GitHub Stars](https://img.shields.io/github/stars/blacksnail789521/Agentic-RL-Training-Recipes?style=social)](https://github.com/blacksnail789521/Agentic-RL-Training-Recipes/stargazers)
![Topic](https://img.shields.io/badge/Agentic%20RL-%20Training%20Recipes-blueviolet)
[![How to Cite](https://img.shields.io/badge/Cite-bibtex-orange)](#citation)

<p align="center"><sub>
✨ If you find our <em>survey</em> useful, a <strong>star ⭐ on GitHub</strong> helps others discover it and keeps you updated on future releases.
</sub></p>

## Table of Contents

- [Training Schemes](#training-schemes)
- [Training Infrastructure](#training-infrastructure)
- [Training Environments](#training-environments)
- [Benchmarks for Training Environments](#benchmarks-for-training-environments)

## Training Schemes
| Paper | TLDR | Component | ↳ Focus | Year | Venue |
| --- | --- | --- | --- | --- | --- |
| [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998) | Iterative self-improvement via generating and filtering high-quality trajectories. | Training Schemes | Rollout & Data Strategy | 2023 | arXiv |
| [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) | Self-play fine-tuning against previous iterations. | Training Schemes | Rollout & Data Strategy | 2024 | ICML |
| [Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](https://arxiv.org/abs/2308.01825) | Simple rejection sampling strategy for data collection. | Training Schemes | Rollout & Data Strategy | 2023 | arXiv |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | Introduces Group Relative Policy Optimization (GRPO) for group-based sampling. | Training Schemes | Rollout & Data Strategy | 2024 | arXiv |
| [Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning](https://arxiv.org/abs/2504.13818) | Down-sampling strategy to filter rollouts and reduce compute. | Training Schemes | Rollout & Data Strategy | 2025 | arXiv |
| [Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2510.24302) | Tree-based exploration with branching at high-uncertainty steps. | Training Schemes | Rollout & Data Strategy | 2025 | arXiv |
| [Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502) | Learn from failure trajectories via contrastive preference pairs. | Training Schemes | Rollout & Data Strategy | 2024 | ACL |
| [SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling](https://arxiv.org/abs/2410.12481) | Adapts soft actor-critic and hindsight replay for open-ended exploration. | Training Schemes | Rollout & Data Strategy | 2024 | arXiv |
| [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](https://arxiv.org/abs/2402.05808) | Reverse curriculum learning starting from goal proximity. | Training Schemes | Rollout & Data Strategy | 2024 | ICML |
| [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Uses predefined progressions in open-ended worlds (Minecraft). | Training Schemes | Rollout & Data Strategy | 2023 | arXiv |
| [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](https://arxiv.org/abs/2511.07317) | Dynamically generates tasks based on current agent performance. | Training Schemes | Rollout & Data Strategy | 2025 | arXiv |
| [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://openreview.net/forum?id=oVKEAFjEqv) | Self-correcting curriculum for web agents. | Training Schemes | Rollout & Data Strategy | 2025 | ICLR |
| [VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2509.19803) | Uses variance of group rewards to prioritize medium-difficulty tasks. | Training Schemes | Rollout & Data Strategy | 2025 | arXiv |
| [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://openreview.net/forum?id=Rwhi91ideu) | Uses exact matching outcome rewards for definitive tasks. | Training Schemes | Feedback & Credit | 2025 | COLM |
| [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) | Exact matching outcome rewards for search tasks. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834) | Functional verification for open-ended intent satisfaction. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005) | Optimizes efficiency by penalizing retrieval costs. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) | Reward modeling for open-ended search tasks. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2506.04185) | Intent satisfaction rewards for search agents. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596) | Optimizes efficiency and functional verification. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [DeepRAG: Thinking to Retrieve Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142) | Optimizes efficiency by explicitly penalizing unnecessary retrieval actions. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [UR^2: Unify RAG and Reasoning through Reinforcement Learning](https://arxiv.org/abs/2508.06165) | Efficiency-aware reward modeling for retrieval. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [ReZero: Enhancing LLM search ability by trying one-more-time](https://arxiv.org/abs/2504.11001) | Rewards focused on relevance and formatting in IR. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146) | IR reward modeling for query diversity and relevance. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2508.20368) | Planning-centric rewards for search agents. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/abs/2505.24332) | IR rewards for deep exploration of topics. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [O^2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering](https://arxiv.org/abs/2505.16582) | Outcome-oriented rewards for search quality. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning](https://arxiv.org/abs/2510.24636) | Tool-augmented reward modeling for long-form agentic tasks. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](https://openreview.net/forum?id=h3LlJ6Bh4S) | Uses Shortest Path Reward Estimation for trajectory quality. | Training Schemes | Feedback & Credit | 2025 | NeurIPS |
| [Synthetic Data Generation and Multi-Step Reinforcement Learning for Reasoning and Tool Use](https://openreview.net/forum?id=oN9STRYQVa) | Holistic history analysis for process rewards. | Training Schemes | Feedback & Credit | 2025 | COLM |
| [Iterative Self-Incentivization Empowers Large Language Models as Agentic Searchers](https://openreview.net/forum?id=s9NkfkUuEr) | Process rewards for search exploration steps. | Training Schemes | Feedback & Credit | 2025 | NeurIPS |
| [ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](https://openreview.net/forum?id=f3sZjkQbv2) | Dense supervision via process reward models (PRM). | Training Schemes | Feedback & Credit | 2025 | NeurIPS |
| [Search and Refine During Think: Facilitating Knowledge Refinement for Improved Retrieval-Augmented Reasoning](https://openreview.net/forum?id=rBlWKIUQey) | Granular evaluation of specific reasoning steps. | Training Schemes | Feedback & Credit | 2025 | NeurIPS |
| [Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation](https://arxiv.org/abs/2510.13272) | Verification-based process rewards for reasoning. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844) | Process supervision for verifiable meta-reasoning steps. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.15107) | Intermediate search quality evaluation. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification](https://arxiv.org/abs/2506.07528) | Hierarchical rewards for information seeking. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning](https://arxiv.org/abs/2508.09303) | Rewards parallel decomposition efficiency. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment](https://arxiv.org/abs/2508.02298) | Uses LLMs to grade/critique intermediate steps. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Advancing Language Multi-Agent Learning with Credit Re-Assignment for Interactive Environment Generalization](https://arxiv.org/abs/2502.14496) | Collaborative grading for UI actions. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic](https://arxiv.org/abs/2511.12159) | LLM-based critique for search trajectories. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](https://openreview.net/forum?id=QAVpe6a3rp) | Verbal feedback for self-correction via retrospective analysis. | Training Schemes | Feedback & Credit | 2025 | NeurIPS |
| [Reflexion: language agents with verbal reinforcement learning](https://openreview.net/forum?id=vAElhFcKW6) | Verbal reinforcement for iterative refinement. | Training Schemes | Feedback & Credit | 2023 | NeurIPS |
| [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2410.01679) | Statistical credit assignment using rollout branching. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Exploiting Tree Structure for Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2509.18314) | Tree-structured estimation for step-wise credit. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732) | Trains dedicated value networks for step influence. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Agentic Reinforcement Learning with Implicit Step Rewards](https://arxiv.org/abs/2509.19199) | Implicit value prediction for self-taught reasoners. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [GRPO-$\lambda$: Credit Assignment improves LLM Reasoning](https://arxiv.org/abs/2510.00194) | Reformulated objective for granular updates without value heads. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) | Group-in-Group policy optimization for credit assignment. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design](https://arxiv.org/abs/2505.11821) | Multi-turn adaptation of GRPO credit assignment. | Training Schemes | Feedback & Credit | 2025 | arXiv |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Standard clipped gradient optimization (Trust Region). | Training Schemes | Policy Optimization | 2017 | arXiv |
| [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118) | Variant of PPO adapted for agentic stability. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | Scaled GRPO for reasoning tasks. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) | Systematized GRPO for scale and distributed training. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [URPO: A Unified Reward & Policy Optimization Framework for Large Language Models](https://arxiv.org/abs/2507.17515) | Unifies policy optimization with reward modeling. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling](https://arxiv.org/abs/2508.17445) | Geometry-aware objectives for tree-structured policies. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning](https://arxiv.org/abs/2508.08221) | Lightweight PPO with entropy bonuses. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems](https://arxiv.org/abs/2504.09037) | Survey on reasoning degeneracy and regularization. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Self-Guided Process Reward Optimization with Redefined Step-wise Advantage for Process Reinforcement Learning](https://arxiv.org/abs/2507.01551) | PRM-free step-level estimation for efficiency. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards](https://arxiv.org/abs/2506.20520) | Asymmetric REINFORCE for off-policy data reuse. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts](https://arxiv.org/abs/2506.02177) | Selective rollout strategy to optimize compute budget. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [A Survey on the Optimization of Large Language Model-based Agents](https://arxiv.org/abs/2503.12434) | Survey on efficient optimization strategies. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning](https://arxiv.org/abs/2502.06781) | Uses early stopping triggers (KL spikes) for stability. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios](https://arxiv.org/abs/2508.17692) | Overview of reasoning stability and hacking. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [AgentBreeder: Mitigating the AI Safety Risks of Multi-Agent Scaffolds via Self-Improvement](https://arxiv.org/abs/2502.00757) | Multi-objective scaffolds for safety and performance. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773) | Constrained MDP formulation for safety. | Training Schemes | Policy Optimization | 2023 | ICLR 2024 |
| [SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](https://arxiv.org/abs/2503.03480) | Safety constraints for vision-language agents. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [Agentic Reinforcement Learning for Search is Unsafe](https://arxiv.org/abs/2510.17431) | Penalizes harmful queries in search agents. | Training Schemes | Policy Optimization | 2025 | arXiv |
| [MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory](https://arxiv.org/abs/2404.11672) | Fine-tunes models to execute explicit read/write API calls. | Training Schemes | Training-Time Memory | 2024 | arXiv |
| [MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation](https://arxiv.org/abs/2308.08239) | Trains multi-stage summarization pipelines. | Training Schemes | Training-Time Memory | 2023 | arXiv |
| [Augmenting Language Models with Long-Term Memory](https://openreview.net/forum?id=BryMFPQ4L6) | Introduces decoupled memory encoders. | Training Schemes | Training-Time Memory | 2023 | NeurIPS |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8) | Trains reflection tokens to trigger on-demand retrieval. | Training Schemes | Training-Time Memory | 2024 | ICLR |
| [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) | Uses RLVR to compress context into a constant footprint. | Training Schemes | Training-Time Memory | 2025 | arXiv |
| [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](https://arxiv.org/abs/2507.02259) | Adapts DAPO for streaming document processing. | Training Schemes | Training-Time Memory | 2025 | arXiv |
| [Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning](https://arxiv.org/abs/2508.19828) | Uses PPO/GRPO to train a dedicated memory-manager agent. | Training Schemes | Training-Time Memory | 2025 | arXiv |
| [LongMemEval: Benchmarking chat assistants on long-term interactive memory](https://arxiv.org/abs/2410.10813) | Benchmark for knowledge updates and abstention. | Training Schemes | Training-Time Memory | 2024 | arXiv |
| [LongBench-v2: Towards deeper understanding and reasoning on realistic long-context multitasks](https://arxiv.org/abs/2410.14481) | Benchmark for extreme-context (2M tokens) tasks. | Training Schemes | Training-Time Memory | 2025 | ACL |


## Training Infrastructure
| Paper | TLDR | Component | ↳ Focus | Link | Year | Venue |
| --- | --- | --- | --- | --- | --- | --- |
| **PPO** | Standard synchronous execution model for on-policy consistency. | Training Infrastructure | Actor-Learner Architectures | | |
| **DeepSeekMath** | Utilizes synchronous execution to simplify credit assignment. | Training Infrastructure | Actor-Learner Architectures | | |
| **AReaL** | Decouples collection and training with staleness control. | Training Infrastructure | Actor-Learner Architectures | | |
| **A3C** | Introduces asynchronous actor-learner decoupling. | Training Infrastructure | Actor-Learner Architectures | | |
| **IMPALA** | V-trace correction for off-policy lag in asynchronous setups. | Training Infrastructure | Actor-Learner Architectures | | |
| **IMPACT** | Uses clipped target networks to stabilize stale critics. | Training Infrastructure | Actor-Learner Architectures | | |
| **LlamaRL** | Orchestrates disjoint GPU groups with AIPO for lag correction. | Training Infrastructure | Actor-Learner Architectures | | |
| **FP16-Precision-RL** | Switches to FP16 to reduce training-inference numerical divergence. | Training Infrastructure | Precision & Acceleration | | |
| **FP8-LM** | Demonstrates doubled throughput with FP8 training. | Training Infrastructure | Precision & Acceleration | | |
| **COAT** | Compresses optimizer states and activations for efficiency. | Training Infrastructure | Precision & Acceleration | | |
| **Smooth-SwiGLU-FP8-Training** | Smoothing operation to handle activation outliers in FP8. | Training Infrastructure | Precision & Acceleration | | |
| **FP4-Training-Framework** | Enables FP4 training via block-wise quantization. | Training Infrastructure | Precision & Acceleration | | |
| **HALO** | Hadamard rotations to spread outliers for low-bit precision. | Training Infrastructure | Precision & Acceleration | | |
| **QeRL** | Uses Adaptive Quantization Noise (AQN) for exploration. | Training Infrastructure | Precision & Acceleration | | |
| **PagedAttention** | Optimized inference kernel that contributes to mismatch. | Training Infrastructure | Training-Inference Mismatch | | |
| **SGLang** | High-throughput inference engine often differing from learner. | Training Infrastructure | Training-Inference Mismatch | | |
| **FSDP** | High-precision training framework (standard baseline). | Training Infrastructure | Training-Inference Mismatch | | |
| **Megatron-LM** | High-precision training framework (standard baseline). | Training Infrastructure | Training-Inference Mismatch | | |
| **MiniMax-M1** | Upcasts output head to FP32 to fix entropy collapse. | Training Infrastructure | Training-Inference Mismatch | | |
| **TIS** | Token-level importance sampling to correct distribution shift. | Training Infrastructure | Training-Inference Mismatch | | |
| **Sequence-Level-Correction** | Sequence masking to robustly handle off-policy shift. | Training Infrastructure | Training-Inference Mismatch | | |
| **DeepSeek-V3.2** | Adopts off-policy sequence masking for training stability. | Training Infrastructure | Training-Inference Mismatch | | |

## Training Environments
| Paper | TLDR | Component | ↳ Focus | Link | Year | Venue |
| --- | --- | --- | --- | --- | --- | --- |
| **ARTIST** | Uses outcome-based rewards to train tool invocation without step-level supervision. | Training Environments | Single-Domain Environments | | |
| **VerlTool** | Modular framework for unified performance across SQL, code, and visual tasks. | Training Environments | Single-Domain Environments | | |
| **TAPO** | Uses Dynamic-PPO to optimize tool usage for external retrieval. | Training Environments | Single-Domain Environments | | |
| **Agent-R1** | Formalizes tool use as an MDP to improve multi-step reasoning. | Training Environments | Single-Domain Environments | | |
| **Simia-RL** | Leverages simulated feedback to overcome data scarcity. | Training Environments | Single-Domain Environments | | |
| **MCP** | Standard for reproducible agent-tool interactions. | Training Environments | Single-Domain Environments | | |
| **Navigating WebAI** | Trains hierarchical T5 planner with V-MPO for web navigation. | Training Environments | Single-Domain Environments | | |
| **AutoWebGLM** | Staged pipeline (SFT $\to$ DPO $\to$ RFT) for stability in web agents. | Training Environments | Single-Domain Environments | | |
| **WebAgent-R1** | Applies on-policy M-GRPO using sparse binary signals in WebArena-Lite. | Training Environments | Single-Domain Environments | | |
| **WebRL** | Uses self-evolving curriculum and outcome reward models for web agents. | Training Environments | Single-Domain Environments | | |
| **ReTool** | Shows reward-driven learning yields emergent tool behaviors. | Training Environments | Single-Domain Environments | | |
| **ToRL** | Demonstrates emergent tool use from reward-driven learning. | Training Environments | Single-Domain Environments | | |
| **One-shot RLVR** | Shows single verifier-rewarded problem can double performance. | Training Environments | Single-Domain Environments | | |
| **StepCoder** | Decomposes code tasks into curriculum-aligned sub-problems with compiler feedback. | Training Environments | Single-Domain Environments | | |
| **BrowserGym** | Aggregates web benchmarks into a fixed schema. | Training Environments | Multi-Domain Environments | | |
| **AgentGym** | Unifies diverse domains via a consistent HTTP interface. | Training Environments | Multi-Domain Environments | | |
| **Mind2Web** | Uses strict splits to penalize layout memorization. | Training Environments | Multi-Domain Environments | | |
| **WebCanvas** | Evaluates agents against live, drifting UIs. | Training Environments | Multi-Domain Environments | | |
| **VisualWebArena** | Unifies multimodal tasks through consistent GUI interfaces. | Training Environments | Multi-Domain Environments | | |
| **WebVoyager** | Unifies multimodal tasks through consistent GUI interfaces. | Training Environments | Multi-Domain Environments | | |
| **AgentGen** | Applies bidirectional evolution to synthesize environment code. | Training Environments | Multi-Domain Environments | | |
| **Eurekaverse** | LLMs write simulation code for embodied control tasks. | Training Environments | Multi-Domain Environments | | |
| **InSTA** | Generates verifiable tasks on unlabeled websites at internet scale. | Training Environments | Multi-Domain Environments | | |
| **Self-Challenging-Agents** | Generates verifiable Code-as-Task instances to bootstrap training data. | Training Environments | Multi-Domain Environments | | |
| **AgentGym-RL** | Dynamically expands horizon from greedy to exploratory settings. | Training Environments | Multi-Domain Environments | | |
| **TTI** | Agents learn to adaptively allocate patience for test-time budgets. | Training Environments | Multi-Domain Environments | | |

## Benchmarks for Training Environments
| Paper | TLDR | Component | ↳ Focus | Link | Year | Venue |
| --- | --- | --- | --- | --- | --- | --- |
| **AgentGym** | Unifies diverse tasks (web, games, databases) under a standard interface. | Benchmarks for Training Environments | Training Gyms | | |
| **BrowserGym** | Unifies diverse web tasks under a standard interface for generalization. | Benchmarks for Training Environments | Training Gyms | | |
| **WebRL** | Provides functional browser environments with self-evolving curricula. | Benchmarks for Training Environments | Training Gyms | | |
| **WebArena** | Functional browser environment for learning navigation policies. | Benchmarks for Training Environments | Training Gyms | | |
| **OSWorld** | High-fidelity environment for computer control (OS). | Benchmarks for Training Environments | Training Gyms | | |
| **AndroidWorld** | High-fidelity environment for mobile control. | Benchmarks for Training Environments | Training Gyms | | |
| **ALFWorld** | Bridges high-level reasoning with low-level embodied physics. | Benchmarks for Training Environments | Training Gyms | | |
| **MMLU** | Standardized exam metrics for general cognitive reasoning. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **GSM8K** | Standardized benchmark for grade-school math reasoning. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **AgentBench** | Quantifies the gap between commercial and open-source models across 8 modalities. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **PaperBench** | Certification test for research reproduction capabilities. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **SWE-bench** | Certification test for software engineering and error recovery. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **CVE-Bench** | Adversarial scenarios to certify robustness against exploits. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **Machiavelli** | Interactive scenarios to certify robustness against ethical hazards. | Benchmarks for Training Environments | Certification Benchmarks | | |
| **R-Judge** | Static trajectories to certify ability to refuse unsafe instructions. | Benchmarks for Training Environments | Certification Benchmarks | | |

## How to Contribute

We welcome contributions! Feel free to open a PR with improvements, fixes, or additional resources.

## Citation

Citation details will be added once the paper is public. Placeholder for now.
```bibtex
@misc{agentic_rl_training_recipes_placeholder,
  title={Training Recipes for Agentic Reinforcement Learning in LLMs: A Survey},
  author={TBD},
  year={TBD},
  note={Citation coming soon}
}
