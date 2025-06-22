# 🧠 Chatbot Arena and Prompt-to-Leaderboard (P2L): A Comprehensive Technical Review

**Author:** Jiaxin Zhang  
*Last Updated: June 2025*

---

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [🥊 Part I: Chatbot Arena – Evaluating with Human Preferences](#-part-i-chatbot-arena--evaluating-with-human-preferences)
  - [🎯 What is Chatbot Arena?](#-what-is-chatbot-arena)
  - [🔧 Evaluation Workflow](#-evaluation-workflow)
  - [📊 Data Snapshot](#-data-snapshot)
  - [📐 Bradley–Terry Model](#-bradleyterry-model)
  - [🆚 BT vs Elo](#-bt-vs-elo)
  - [🔮 Future Directions](#-future-directions)

- [🧾 Part II: Prompt-to-Leaderboard (P2L) – Personalized Rankings](#-part-ii-prompt-to-leaderboard-p2l--personalized-rankings)
  - [🎯 Importance of P2L](#-importance-of-p2l)
  - [🧱 Architecture Overview](#-architecture-overview)
  - [🔍 Pipeline Construction](#-pipeline-construction)
  - [📤 Routing Decision Logic](#-routing-decision-logic)
  - [📏 Results Summary](#-results-summary)
  - [📊 P2L Evaluation Metrics](#-p2l-evaluation-metrics)
  - [💡 Applications](#-applications)

- [🔗 Additional Resources](#-additional-resources)
  
---

## 📌 Overview

Large Language Models (LLMs), such as GPT-4, Claude, and Gemini, are rapidly advancing. However, evaluating them with human-aligned and scalable frameworks remains an open challenge. LMSYS addresses this with:

- **Chatbot Arena** – A crowdsourced preference-based evaluation platform.
- **Prompt-to-Leaderboard (P2L)** – A system to route prompts to the best model and generate personalized leaderboards.

---

## 🥊 Part I: Chatbot Arena – Evaluating with Human Preferences

### 🎯 What is Chatbot Arena?

Chatbot Arena is a **scalable, model-agnostic, and preference-based evaluation** system. It crowdsources real human judgments via anonymous model pair comparisons, simulating real-world user experience.

---

### 🔧 Evaluation Workflow

1. Users submit a prompt.
2. Two models (Model A and B) generate answers anonymously.
3. Users compare responses and vote for the better one.
4. Models are shuffled to avoid bias.
5. Votes are logged for statistical ranking.

---

### 📊 Data Snapshot

| Metric           | Value        |
|------------------|--------------|
| Total Votes      | 2.8M+        |
| Unique Prompts   | 300,000+     |
| Models Compared  | 219+         |
| Domains          | Math, Code, Dialogue, Reasoning, Writing |

---

### 📐 Bradley–Terry Model

Arena uses the **Bradley–Terry (BT)** model to infer global rankings from pairwise human preferences.

#### 📐 Bradley–Terry Model

The Bradley–Terry (BT) model estimates a global ranking from pairwise human preferences.

##### Probability Model

Given two models $i$ and $j$, the probability that model $i$ is preferred over model $j$ is:

$$
P(i > j) = \frac{1}{1 + \exp(\xi_j - \xi_i)}
$$

where $\xi_i$ and $\xi_j$ are the latent skill scores associated with models $i$ and $j$, respectively.

##### Likelihood Function

The total log-likelihood over all pairwise comparisons is:

$$
\mathcal{L}(\xi) = \sum_{i \ne j} n_{ij} \cdot \log \left( \frac{1}{1 + \exp(\xi_j - \xi_i)} \right)
$$

- $n_{ij}$ : the number of times model $i$ beats model $j$ 
- Parameters $\xi$ are estimated via **Maximum Likelihood Estimation (MLE)**

This framework enables robust estimation of model rankings even in the presence of sparse or asymmetric comparison data.


### 🆚 BT vs Elo

| Feature             | Bradley–Terry (BT)     | Elo                      |
|---------------------|-------------------------|---------------------------|
| Data Requirements   | Sparse, asymmetric      | Repeated symmetric games |
| Stability           | High (MLE-based)        | Medium (Online updates)  |
| Transitive Inference| Yes                     | Limited                  |
| Use Case Fit        | Arena (LLM outputs)     | Chess, Go, etc.          |

**Bottom line**: BT better captures LLM evaluation complexities.

---



## 🔮 Future Directions

As the Arena and P2L systems mature, several promising directions are emerging to improve personalization, fairness, and adaptability:

- **Cluster-Aware BT Models:**  
  Adapt the Bradley–Terry model to account for clusters of users with distinct preferences, allowing more nuanced inference.

- **User-Customized Leaderboards:**  
  Generate leaderboard views tailored to individual or grouped user behavior, preferences, and task domains.

- **Preference-Aligned Deployments:**  
  Deploy LLMs that dynamically adapt to user intent — factual, creative, empathetic — by routing requests to the most aligned model.

These innovations pave the way for truly user-centric and interpretable AI evaluation.


## 🧾 Part II: Prompt-to-Leaderboard (P2L) – Personalized Rankings

### 🎯 Importance of P2L

Different LLMs excel in different areas, making **prompt-specific evaluation and routing** critical:

- **GPT-4**: Strong in mathematics and symbolic reasoning  
- **Claude-3**: Excels at logical reasoning and subtle alignment tasks  
- **Gemini**: Performs best on multimodal and vision-related prompts  

**Goal**:  
Leverage prompt-level evaluation to:
- Dynamically **route each prompt to the most suitable model**  
- Construct **personalized, domain-aware rankings**  
- Improve overall system performance beyond what any single model can offer  

---

### 🧱 Architecture Overview

The Prompt-to-Leaderboard (P2L) architecture is designed to learn user-aligned model preferences.

**Input**:
- Prompt $Z$
- Model encoding vector $X \in \{-1, +1\}$, where -1 represents Model A and +1 represents Model B

**Output**:
- Preference label $Y \in \{0, 1\}$, indicating which model the human preferred

Model predicts preference probability $\hat{y}$ via a **sigmoid head** on an LLM embedding of the prompt:

$$
\hat{y} = \sigma \left( X^\top \hat{\theta}(Z) \right)
$$

- Use LoRA-tuned LLMs (e.g., Qwen2.5-1.5B)
- Train only the final preference head on top of the frozen LLM encoder

---

### 🔍 Pipeline Construction

#### 1. Data

- Source: **Arena-55K** dataset  
- Format: $(Z, A(Z), B(Z), Y)$, where:
  - $Z$: Prompt  
  - $A(Z), B(Z)$: Model A and B responses  
  - $Y$: Human preference label

#### 2. Training

- Loss function: **Binary Cross-Entropy**

$$
\mathcal{L} = - y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
$$

- Hyperparameters:
  - Batch size: 4  
  - Max sequence length: 4096  
  - Optimizer: Adam  
  - Learning rate: $1 \times 10^{-5} \sim 5 \times 10^{-5}$  

#### 3. Inference

- For a new prompt $Z$, the model predicts preference probabilities for any pair of models $(i, j)$  
- Construct a **pairwise win probability matrix**:

$$
P_{i > j}(Z) = \Pr(\text{Model } i \text{ preferred over } j \mid Z)
$$

---

### 📤 Routing Decision Logic

Use the predicted pairwise preference matrix to **compute the optimal routing strategy**.

Goal:  
Find the optimal **mixed strategy** $\pi^* \in \Delta_M$ (a distribution over $M$ models):

$$
\pi^* = \arg\max_{\pi \in \Delta_M} \sum_{i, j} \pi_i \cdot P(i > j) \cdot q_j
$$

Where:
- $\pi_i$: probability of routing to model $i$  
- $P(i > j)$: model $i$'s win rate over model $j$ on prompt $Z$  
- $q_j$: importance or prior weight for model $j$

---

### 🔢 Algorithms for Computing $\pi^*$

| **Component**               | **Description**                                                                                                                                                               |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Inputs**                 | - $q$: Model weight vector  <br> - $W^\ast$: Pairwise win-rate matrix  <br> - $\theta^\ast(z)_j$: BT score of model $j$ on prompt $z$  <br> - $c$: Per-model cost vector  <br> - $C$: Total cost budget |
| **Optimization Objective** | Solve the linear program to compute the optimal routing distribution:  <br>  $\tilde{\pi}^\ast = \arg\max_{\tilde{\pi} \in \Delta_M,\ \tilde{\pi}^\top c \leq C} \tilde{\pi}^\top W^\ast q$ |
| **Reward Computation**     | Compute the expected reward for the optimal routing:  <br> $R^\ast = \tilde{\pi}^{\ast\top} W^\ast q$ |
| **BT Score Estimation**    | Estimate the router’s equivalent BT score $\theta'$ by solving:  <br> $\sum_a q_a \cdot \sigma(\theta' - \theta^\ast(z)_a) = R^\ast$  <br> where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function |
| **Outputs**                | - $\tilde{\pi}^\ast$: Optimal routing distribution  <br> - $\theta'$: Estimated BT score for the router |


---


These algorithms allow for **optimal prompt-to-model assignment** in real-time, balancing accuracy, diversity, and user satisfaction.

---


### 📊 P2L Evaluation Metrics

- **Local Accuracy**: Binary correctness on pairwise vote prediction
- **Log Loss**: Model confidence calibration
- **Top-k Precision**: Ranking accuracy at leaderboard head
- **Kendall’s τ / Spearman’s ρ**: Rank correlation with ground truth
- **Spread**: Variation in P2L score across prompts (used to filter ambiguous queries)

---

### 📏 Results Summary

| Task Domain        | Best Individual Model | P2L Routed Accuracy |
|--------------------|------------------------|----------------------|
| General Tasks      | GPT-4                  | 76.7%                |
| Mathematics        | GPT-4                  | 78.3%                |
| Programming        | Claude-3               | 81.2%                |
| Logical Reasoning  | GPT-4                  | 75.5%                |

---

### 💡 Applications

Prompt-level evaluation enables fine-grained control, diagnostic power, and personalization across LLM workflows. Below are major application areas:

- 🔁 **Optimal Routing**  
  Use per-prompt scores to automatically select the best model for each input.  
  - Dynamically dispatch prompts based on past performance.  
  - Power ensemble systems where different models specialize in different tasks.  
  - Improve multi-agent frameworks and chat systems through adaptive routing.  

- 👤 **Personalized Evaluation**  
  Build custom rankings based on individual user preferences and behaviors.  
  - Track user-specific feedback across prompt types.  
  - Enable user profiles that evolve with usage.  
  - Prioritize models that align with a user’s domain (e.g., legal, technical, creative).  

- 🩺 **Automated Diagnosis**  
  Analyze model performance on specific prompt types or domains to find where it struggles (blind spots) and where it excels (strengths). 
  - Detect patterns of failure (e.g., hallucination, inconsistency).  
  - Visualize model performance heatmaps across topics or difficulty.  
  - Inform targeted fine-tuning and data augmentation strategies.  
---

## 🔗 Additional Resources

- 📘 [Chatbot Arena Paper](https://arxiv.org/abs/2403.04132)
- 📘 [P2L Paper](https://arxiv.org/abs/2502.14855)
- 🧠 [Chatbot Arena Website](https://chat.lmsys.org/)
- 📂 [P2L GitHub](https://github.com/lmarena/p2l)
- 📑 [P2L Slides](./p2l_slides.pdf)

