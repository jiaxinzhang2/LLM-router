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

#### Probability a model \( i \) wins over model \( j \):

```
P(i > j) = 1 / (1 + exp(ξ_j - ξ_i))
```

Where \( ξ_i \) is the skill score of model \( i \).

#### Likelihood Function:

```
L(ξ) = Σ_{i≠j} n_ij * log(1 / (1 + exp(ξ_j - ξ_i)))
```

- \( n_{ij} \): number of times \( i \) beats \( j \)
- Solved via Maximum Likelihood Estimation (MLE)

---

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

Different models shine in different areas:

- GPT-4: Mathematics
- Claude-3: Logical Reasoning
- Gemini: Multimodal tasks

**Goal**: Route prompts to the optimal model and build dynamic rankings.

---

### 🧱 Architecture Overview

**Input**: Prompt \( Z \), Model Encoding \( X \) (-1 for A, +1 for B)\
**Output**: Preference label \( Y \in \{0, 1\} \)

- Use LoRA-tuned LLMs (e.g., Qwen2.5-1.5B)
- Train to predict preference probability via sigmoid head

```
ŷ = σ(Xᵀ θ̂(Z))
```

---

### 🔍 Pipeline Construction

#### 1. Data

- Arena-55K samples
- Format: (Prompt, Model A, Model B, Vote)

#### 2. Training

- Binary Cross-Entropy Loss:
```
L = - y * log(ŷ) - (1 - y) * log(1 - ŷ)
```

- Batch size: 4
- Max seq length: 4096
- Optimizer: Adam
- LR: 1e-5 ~ 5e-5

#### 3. Inference

- For prompt \( Z \), predict preference between any two models \( (i, j) \)
- Construct matrix of pairwise win probabilities

---

### 📤 Routing Decision Logic

- Compute optimal mixed strategy \( \pi^* \in \Delta_M \)

```
π* = argmax_π Σ_{i,j} π_i * P(i > j) * q_j
```

Where:
- \( π \): model deployment probabilities
- \( q_j \): weights for model \( j \)

**Algorithms**:
- Borda Count
- Expected Rank Minimization
- Plackett–Luce Sampling

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

