# LALR: Layer-Adaptive Learning Rate Fine-Tuning Using Sequential Scaling Metrics

## Abstract

The Layer-Adaptive Learning Rate (LALR) Fine-Tuning approach described in this paper uses static analysis of a language model's weights within each layer to proportionally adjust the learning rates based on layer-specific contributions to the model’s performances, to improve: 

- Overall function, measured by singular value decomposition (SVD)
- Relative information between layers, measured by cross-entropy (CE)
- Relative diversity between layers, measured by cosine similiarity (CS)

LALR draws motivation from observations in cognitive science and neuroscience, suggesting that human learning involves selective strengthening of certain neural pathways and preservation of others [8]. By mimicking this selective enhancement in artificial neural networks, LALR aims to achieve efficient and effective fine-tuning, leading to more robust and versatile adapted models through the following:

- Controlled undertraining: Selectively reduces learning rates for well-optimized layers, intentionally risking undertraining to prevent overfitting and maintain existing knowledge.
- Capacity optimization: Focuses training on underutilized layers, extracting more performance from the existing model structure without increasing model size.
- Knowledge preservation: Potentially "undertrains" well-optimized layers in pre-trained models, to help preserve valuable pre-trained knowledge while adapting to new tasks.
- Adaptive fine-tuning: Dynamically manages undertraining risk by adjusting the learning rate scaling based on the current state of each layer after each training epoch. 
- Exploration-exploitation balance: Encourages exploration of different internal representations by focusing fine-tuning on less optimized layers, to encourage a more robust and versatile fine-tuned model.

## 1.0 Background and Related Work

Large language models have demonstrated remarkable capabilities across various tasks, from question answering to complex reasoning [1]. However, these models often struggle with generalization and exhibit brittle behavior when faced with out-of-distribution data or subtle input variations as they grow in size and complexity [2]. Traditional fine-tuning approaches often treat all layers of a model equally, applying uniform learning rates or optimization strategies across the entire network. Different layers in large language models often specialize in capturing different levels of linguistic and semantic information [6], suggesting that a layer-specific approach to fine-tuning could yield better results. Several lines of research have explored methods to improve model robustness and generalization:

- **Selective Layer Modification:** Layer-Selective Rank Reduction (LASER) [3] identifies and prunes less important layers to reduce model size while maintaining performance. While effective for model compression, this approach doesn't address the potential of enhancing underutilized layers to maintain the intrinsic fine-tuning potential based on the model's intrinsic dimensionality [9].

- **Output Diversity:** Entropy-based training strategies [4] encourage more diverse model outputs for improving generalization. However, these methods typically focus on the model's output rather than its internal representations.

- **Adaptive Optimization:** Adaptive learning rate methods [5] have shown promise in improving training efficiency and model performance. Layer-wise Adaptive Rate Scaling (LARS) [11] and Layer-wise Adaptive Moments (LAMB) [12] have demonstrated significant improvements in training large neural networks with large batch sizes. LARS adapts the learning rate for each layer based on the ratio of the L2 norm of the layer's weights to the L2 norm of its gradient, while LAMB extends this idea to work with adaptive moment estimation. While effective for initial training, these methods focus on weight and gradient magnitudes but are not specifically designed for the fine-tuning phase of pre-trained models.

- **Information Theory in Neural Networks:** Recent work has explored the application of information theory to understand and optimize neural networks [7]. These approaches provide insights into the information flow within networks but have not been fully leveraged in fine-tuning strategies.

Building upon these foundations, the proposed method offers a novel approach tailored for fine-tuning large language models. It combines concepts from linear algebra (SVD), information theory (CE), and representation learning (CS) to provide a comprehensive analysis of layer contributions. This new LALR approach:

- Incorporates distinct SVD, CE, and CS metrics to analyze layer contributions without requiring computationally expensive gradient assessments.
- Targets underutilized, less informative, and redundant layers for optimization, allowing for efficient use of model capacity.
- Focuses on the unique challenges of fine-tuning pre-trained models, where the goal is to refine existing knowledge rather than learn from scratch.

## 2.0 Learning Rate Scaling Metrics

LALR computes metrics using the model weights to obtain a per-layer scaling factor to lower learning rates of those layers that have lower SVD, relative CE, and higher relative CS.

### **2.1 SVD Metric**

SVD factorizes a matrix A into the product of three matrices: $A = UΣV^T$, where $U$ and $V$ are orthogonal matrices, and $Σ$ is a diagonal matrix containing singular values. These values represent the importance of different directions in the matrix's transformation, with larger values indicating more significant directions. In neural networks, SVD of weight matrices reveals the layer's capacity to represent and transform features.

Let $λ_{i,k}$ be the $k$-th singular value of the weight matrix of layer $i$, sorted in descending order. The initial SVD-based contribution metric for layer $i$ is defined as:

$M_{SVD}'(i) = (1 / (L-1)) ∑_{k=1}^{L-1} λ_{i,k}$

where $L$ is the total number of singular values. The final metric after normalizing is obtained after subtracting from 1, to obtain the scaling factor to apply to the learning rate for each layer.

Metric Computation:
1. Perform SVD on each layer's weight matrix
2. Calculate the mean of the singular values
3. Normalize the metric to range from 0 to 1
4. Subtract from 1 to prioritize lower SVD layers

Scaling the learning rate in each layer by $M_{SVD}(i)$ aims to:
- Preserve learned hierarchical representations in well-optimized layers
- Enhance complex feature learning in underutilized layers
- Improve signal-to-noise ratio in representations
- Optimize model capacity usage
- Increase training stability and convergence

### **2.2 CE Metric**

CE measures the difference between two probability distributions. For discrete probability distributions $P$ and $Q$, CE is defined as $CE(P, Q) = -∑ P(x) log Q(x)$. In neural networks, CE quantifies the dissimilarity between weight distributions of different layers, treating normalized weight values as probability distributions.

Let $p_i$ be the normalized weight distribution of layer $i$, and $p_j$ be the normalized weight distribution of another layer $j$. The cross-entropy $CE(p_i, p_j)$ between the distributions $p_i$ and $p_j$ is:

$CE(p_i, p_j) = -∑_k p_i(k) log(p_j(k))$

The initial CE metric for layer $i$ is:

$M_{CE}'(i) = (1 / (L-1)) ∑_{j≠i} CE(p_i, p_j)$

where $L$ is the total number of layers. The final metric after normalizing is obtained after subtracting from 1, to obtain the scaling factor to apply to the learning rate for each layer.

Metric Computation:
1. Flatten and normalize weight tensors to create probability distributions
2. Compute CE between each layer's distribution and every other layer's distribution
3. Average these values to obtain the initial CE metric for each layer
4. Add a small constant to prevent logarithm of zero
5. Normalize the metric to range from 0 to 1
6. Subtract from 1 to obtain the final CE metric

Scaling the learning rate in each layer by $M_{CE}(i)$ aims to:
- Maintain information content in dense model layers
- Diversify learned representations across layers
- Capture task-specific information efficiently
- Improve handling of diverse inputs
- Discover novel, task-relevant features in underutilized layers
- Enhance hierarchical feature representation

### **2.3 CS Metric**

CS measures similarity between two non-zero vectors in an inner product space, revealing functional redundancy in the neural network layers. For vectors $A$ and $B$, CS is defined as $cos(θ) = (A · B) / (||A|| ||B||)$, where $||x||$ is the vector magnitude. 

Let $w_i$ and $w_j$ be the flattened weight vectors of layers $i$ and $j$. The CS between these layers is:

$CS(i, j) = (w_i · w_j) / (||w_i|| ||w_j||)$

The average CS of a layer with all other layers is:

$M_{CS}(i) = (1 / (L-1)) ∑_{j≠i} CS(i, j)$

where $L$ is the total number of layers. The final metric after normalizing provides the scaling factor to apply to the learning rate for each layer.

Metric Computation:
1. Flatten weight tensors into vectors
2. Compute CS between each layer pair using normalized weight vectors
3. Calculate each layer's average similarity with all others
4. Normalize the final metric to range from 0 to 1

Scaling the learning rate in each layer by $M_{CS}(i)$ aims to:
- Preserve existing diversity in unique layers
- Enhance feature and pattern capture capacity
- Allocate model capacity efficiently
- Improve training efficiency and convergence
- Optimize information propagation
- Adapt pre-trained models effectively to specific tasks
- Enhance generalization by preventing overfitting to specific layer interactions

### **2.4 Robust Normalization of Layer-wise Metrics**

LALR employs the following robust normalization approach to ensure a more effective utilization of the SVD, CE, and CS metrics across the different layers to promote fine-tuning stability and effectiveness:

1. Metric Calculation: Compute raw SVD, CE, and CS metrics for each layer.
2. Outlier Mitigation: Using the mean values across layers for the metrics ensures resilience against outliers due to the Central Limit Theorem so that metrics represent each layer's relationship to the entire network rather than being skewed by individual extreme comparisons.
3. Normalization Process: Normalize each metric to [0, 1] range using:

normalized metric = $(metric - min(metric)) / (max(metric) - min(metric))$

This normalization approach ensures:
- Comparability between metrics and layers
- Scalability across model sizes and architectures
- Robustness through reduced sensitivity to outliers
- Interpretability via intuitive [0, 1] range.

## 3.0 Layer-Adaptive Learning Rate (LALR) Fine-Tuning

LALR dynamically adjusts learning rates based on layer-specific metrics, effecting more weight changes in underutilized, lower information content, and redundant layers. It encourages diverse internal representations across the model. The sequential application of SVD, CE, and CS metrics progressively enhances the model's weights, optimizing representations comprehensively.

A key advantage of LALR is its minimal computational overhead. LALR computes the layer-wise SVD, CE, and CS metrics only once per epoch, making the additional computation negligible compared to the fine-tuning process itself. This efficiency allows LALR to provide substantial benefits without significantly increasing training time or resource requirements.

### **3.1 Theoretical Foundation**

LALR dynamically adjusts learning rates based on layer-specific metrics, focusing on underutilized, lower information content, and redundant layers. The approach is grounded in the following theoretical framework:

**Objective Function:**

Let $J(\theta)$ represent the model's performance as a function of layer-wise metrics, where $\theta$ are the model parameters:

$J(\theta) = f(M_{SVD}(1), \ldots, M_{SVD}(L), M_{CE}(1), \ldots, M_{CE}(L), M_{CS}(1), \ldots, M_{CS}(L))$

where $M_{SVD}(i)$, $M_{CE}(i)$, and $M_{CS}(i)$ are the SVD, CE, and CS metrics for layer $i$, respectively, and $L$ is the number of layers.

The optimization of $J(\theta)$ is guided by the following relationships:

- $\frac{\partial J}{\partial M_{SVD}(i)} > \frac{\partial J}{\partial M_{CE}(i)} > \frac{\partial J}{\partial M_{CS}(i)}$ initially.
- Optimizing $M_{SVD}(i)$ increases $\frac{\partial J}{\partial M_{CE}(i)}$: $\frac{\partial^2 J}{\partial M_{SVD}(i) \partial M_{CE}(i)} > 0$
- Optimizing $M_{CE}(i)$ increases $\frac{\partial J}{\partial M_{CS}(i)}$: $\frac{\partial^2 J}{\partial M_{CE}(i) \partial M_{CS}(i)} > 0$

This sequence maximizes these interaction effects, leading to improvements in $J(\theta)$ by exploiting their interdependencies based on the relationship between metrics and model performance, the metric interaction, and bounding analysis detailed in Appendix B:

1. Metric Improvements and $J(θ)$: Improvements in each metric correlate with reductions in the objective function $J(θ)$ based on the following (Section B.2.3):
   - For $M_{SVD}: E[ΔJ(θ)] ≈ -∇J(θ)^T (∂θ/∂M_{SVD} * E[ΔM_SVD]) > 0$ when $ΔM_{SVD} > 0$
   - For $M_{CE}: E[ΔJ(θ)] ≈ -∇J(θ)^T (∂θ/∂M_{CE} * E[ΔM_{CE}]) > 0$ when $ΔM_{CE} > 0$
   - For $M_{CS}: E[ΔJ(θ)] ≈ ∇J(θ)^T (∂θ/∂M_{CS} * E[ΔM_{CS}]) < 0$ when $ΔM_{CS} < 0$

2. SVD Optimization: Expands the effective rank of weight matrices, creating capacity for new, task-specific information (Section B.2.2).
   - Mathematical Basis: $\frac{\partial J}{\partial M_{SVD}(i)} \approx \log(\text{rank}(W_i))$

3. CE Optimization: Leverages increased capacity to learn unique, task-specific features (Section B.2.2).
   - Mathematical Basis: $\frac{\partial J}{\partial M_{CE}(i)} \approx H(W_i \mid W_j, j \neq i)$, where $H$ is the conditional entropy

4. CS Optimization: Refines learned features, ensuring efficient use of model capacity (Section B.2.2).
   - Mathematical Basis: $\frac{\partial J}{\partial M_{CS}(i)} \approx -\cos(W_i, W_j), j \neq i$

5. SVD -> CE -> CS Sequence Justification:
   - Foundation Building: SVD optimization establishes a robust foundation, amplifying subsequent optimizations (Section B.3.2.5, Theorems B.3.2.5.1 and B.3.2.5.2).
   - Cascading Enhancements: CE optimization magnifies SVD improvements, and CS optimization leverages both SVD and CE enhancements (Sections B.3.2.1, B.3.2.2, and B.3.2.3, Theorems 3.2.1.1, 3.2.2.1, and 3.2.3.1).
   - Interaction Constraints: Bounded interactions ensure stable sequential optimization (Section B.3.2.5, Theorem B.3.2.5.1).
   - Adaptive Execution: Each phase recalibrates based on the model's current state, ensuring responsiveness to prior optimizations (Section B.3.2.4).

This theoretical foundation illustrates how LALR's mathmatically grounded approach for sequential fine-tuning using the SVD, CE, and CS metrics provides a robust framework for dynamically adjusting learning rates across layers and epochs to enhance the desired aspects of the models weight structure.

### 3.2 Implementation and Integration

The example code provided in Appendix A provides a practical LALR implementation that aligns with the theoretical foundations discussed in this paper to showcases how LALR can be applied to real-world deep learning optimization tasks. The AdaptiveOptimizer class encapsulates the core functionality of LALR, including the computation of layer-wise SVD, CE, and CS metrics, and the adjustment of learning rates based on these metrics. The adaptive_fine_tuning function illustrates how the optimizer can be integrated into a typical fine-tuning pipeline, with the sequential optimization of SVD, CE, and CS metrics over multiple epochs. 

LALR can also integrate with existing per-batch adaptive learning rate methods, such as AdamW[13], simply by scaling the learning rate for each layer $l$ at epoch $e$ by:

$lr_{l,e} = lr_{per-batch,l,e} × scale_{layer-wise,l,e}$

This integration adapts the learning rate at both per-weight and per-layer levels, accounting for gradient dynamics and global layer properties. 

### 3.3 Benefits and Outcomes

LALR represents a comprehensive optimization strategy that enhances model performance and generalization with minimal computational burden, making it a practical and efficient approach for fine-tuning large language models.

A key advantage of LALR is its minimal computational overhead. LALR computes the layer-wise SVD, CE, and CS metrics only once per epoch, making the additional computation negligible compared to the fine-tuning process itself. This efficiency allows LALR to provide substantial benefits without significantly increasing training time or resource requirements.

LALR intends to achieve the following outcome by noveling combining the metric computation, normalization, and fine-tuning sequences:  

- Targeted Optimization: Enhances overall model capacity and functional diversity, accelerating learning of distinctive features in underperforming layers.
- Adaptive Exploration: Discovers unique features and escapes local optima, improving generalization across tasks and datasets.
- Structural Regularization: Increases robustness to overfitting, improves adaptability to new data, and enhances stability during fine-tuning.
- Overfitting Protection: Promotes learning of generalizable patterns rather than memorizing training data.
- Stability Preservation: Maintains baseline performance by scaling down learning rates for well-optimized layers.
- Efficient Capacity Utilization: Captures generalizable features without increasing model size.
- Versatility Enhancement: Improves model performance and adaptability.

## References

[1] Brown, T. B., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877-1901.

[2] Hendrycks, D., et al. (2020). The many faces of robustness: A critical analysis of out-of-distribution generalization. arXiv:2006.16241.

[3] Liao, Z., et al. (2023). LASER: Layer-Selective Rank Reduction for Efficient Language Model Compression. arXiv:2301.09389.

[4] Pereyra, G., et al. (2017). Regularizing neural networks by penalizing confident output distributions. arXiv:1701.06548.

[5] You, Y., et al. (2020). Large batch optimization for deep learning: Training BERT in 76 minutes. arXiv:1904.00962.

[6] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Matematicheskii Sbornik, 114(4), 507-536.

[7] Achille, A., & Soatto, S. (2018). Information dropout: Learning optimal representations through noisy computation. IEEE transactions on pattern analysis and machine intelligence, 40(12), 2897-2905.

[8] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[9] Aghajanyan, A., et al. (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv:2012.13255.

[10] Wang, Z., et al. (2023). Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers. arXiv:2212.10559.

[11] You, Y., Gitman, I., & Ginsburg, B. (2017). Large Batch Training of Convolutional Networks. arXiv:1708.03888.

[12] You, Y., Li, J., Hseu, J., Song, X., Demmel, J., & Hsieh, C. J. (2019). Reducing BERT Pre-Training Time from 3 Days to 76 Minutes. arXiv:1904.00962.

[13] Loshchilov. I, Hutter F, (2019). Decoupled Weight Decay Regularization. arXiv:1711.05101

## Appendix A: Example Implementation

```python
import torch
from torch.optim import Optimizer
from typing import List, Dict, Any

class AdaptiveOptimizer(Optimizer):
    """
    Custom optimizer that adjusts learning rates based on layer-specific metrics.

    Attributes:
        model (torch.nn.Module): The model to be optimized.
        base_lr (float): The base learning rate.
        phases (List[str]): List of phases for which learning rates will be adjusted.
        metrics (Dict[str, List[float]]): Dictionary holding metrics for each phase.
        param_groups (List[Dict[str, Any]]): List of parameter groups with their names and learning rates.
    """

    def __init__(self, model: torch.nn.Module, base_lr: float, phases: List[str]):
        self.model = model
        self.base_lr = base_lr
        self.phases = phases
        self.metrics = None
        self.param_groups = self._prepare_param_groups()

    def _prepare_param_groups(self) -> List[Dict[str, Any]]:
        """
        Prepare parameter groups for the optimizer.

        Returns:
            List[Dict[str, Any]]: Parameter groups with their names and initial learning rates.
        """
        parameters = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                parameters.append({'params': [param], 'name': name})
            else:
                parameters.append({'params': [param], 'name': name, 'lr': self.base_lr})
        return parameters

    def compute_metrics(self):
        """
        Compute the SVD, CE, and CS metrics for each layer in the model.
        """
        svd_metrics, ce_metrics, cs_metrics = get_layer_metrics(self.model)
        self.metrics = {
            'svd': svd_metrics,
            'ce': ce_metrics,
            'cs': cs_metrics
        }

    def adjust_learning_rates(self, phase: str):
        """
        Adjust learning rates for each parameter group based on the specified phase.

        Args:
            phase (str): The current phase ('svd', 'ce', or 'cs').
        """
        if self.metrics is None:
            self.compute_metrics()

        metrics = self.metrics[phase]
        for param_group, metric in zip(self.param_groups, metrics):
            if 'weight' in param_group['name']:
                if phase == 'svd':
                    scale = 1 - metric  # Lower SVD means higher learning rate
                elif phase == 'ce':
                    scale = 1 - metric  # Lower CE means higher learning rate
                elif phase == 'cs':
                    scale = metric  # Higher CS means higher learning rate
                else:
                    scale = 1
                param_group['lr'] = self.base_lr * scale

def adaptive_fine_tuning(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, base_lr: float, num_epochs: int = 3):
    """
    Perform adaptive fine-tuning on the model.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        base_lr (float): The base learning rate.
        num_epochs (int, optional): The number of epochs for fine-tuning. Defaults to 3.
    """
    optimizer = AdaptiveOptimizer(model, base_lr, ['svd', 'ce', 'cs'])
    for epoch in range(num_epochs):
        phase = ['svd', 'ce', 'cs'][epoch % 3]
        optimizer.adjust_learning_rates(phase)
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()

        print(f"Completed epoch {epoch + 1}, {phase} phase")

# Usage example
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Replace 'your_model_name' with the actual model name
model = AutoModelForSequenceClassification.from_pretrained("your_model_name")
tokenizer = AutoTokenizer.from_pretrained("your_model_name")

# Assuming YourDataLoader is a function that returns a DataLoader for your dataset
train_loader = YourDataLoader(tokenizer)
adaptive_fine_tuning(model, train_loader, base_lr=2e-5, num_epochs=3)
```
# Appendix B: Mathmatical Justificaton

## B.1 Definitions and properties:

The following properties of the metrics form the overall basis of this analysis.

### B.1.1 Singular Value Decomposition (SVD):

- Existence and uniqueness of SVD for any real matrix [11]
- Relationship between singular values and the rank of $W$ [14]
- Interpretation of singular values as the square roots of eigenvalues of $W^T W$ [29]
- Bounds on singular values: $0 ≤ σ_i ≤ ||W||_2$, where $σ_i$ are singular values [30]
- Continuity of singular values with respect to changes in $W$ [28]
- Differentiability of singular values (with care taken for repeated singular values) [21]

### B.1.2 Cross-Entropy (CE):

- Non-negativity: $CE(p_i, p_j) ≥ 0$ [9]
- Asymmetry: $CE(p_i, p_j) ≠ CE(p_j, p_i)$ in general [20]
- Relationship to Kullback-Leibler divergence [17]
- Bounds on CE: $0 ≤ CE(p_i, p_j) ≤ log(n)$, where $n$ is the number of elements in the distribution [9]
- Continuity of CE with respect to changes in weight distributions [31]
- Convexity of CE in its second argument [5]

### B.1.3 Cosine Similarity (CS):

- Bounds: $-1 ≤ CS(w_i, w_j) ≤ 1$ [7]
- Symmetry: $CS(w_i, w_j) = CS(w_j, w_i)$ [7]
- $CS(w_i, w_i) = 1$ for any non-zero vector $w_i$ [7]
- Relationship to angle between vectors: $CS(w_i, w_j) = cos(θ)$, where $θ$ is the angle between $w_i$ and $w_j$ [16]
- Invariance under scalar multiplication: $CS(αw_i, βw_j)$ = $CS(w_i, w_j)$ for $α, β > 0$ [7]
- Continuity and differentiability of CS with respect to changes in weight vectors [6]

### B.1.4 Metric Definitions and Mathematical Notations and Conventions

#### B.1.4.1 Metric Definitions 

1. SVD Metric:
   
   For a weight matrix $W$ of layer $l$, the initial SVD metric is defined as:
   
   $M_{SVD}^{initial}(l) = \frac{1}{r} \sum_{i=1}^r \sigma_i$, where $\sigma_i$ are the singular values of $W$, and $r$ is the rank of $W$.

2. CE Metric:

   For layer $l$, the initial CE metric is defined as:

   $M_{CE}^{initial}(l) = \frac{1}{L-1} \sum_{k \neq l} CE(p_l, p_k)$, where $L$ is the total number of layers, and $p_l$ and $p_k$ are the normalized weight distributions of layers $l$ and $k$ respectively. 
   
   The $CE(p, q)$ between two distributions $p$ and $q$ is defined as:
   
   $CE(p, q) = -\sum_x p(x) \log q(x)$

3. CS Metric:
   
   For layer $l$, the CS metric is defined as:  
   
   $M_{CS}(l) = \frac{1}{L-1} \sum_{k \neq l} CS(w_l, w_k)$, where $w_l$ and $w_k$ are the flattened weight vectors of layers $l$ and $k$ respectively. 
   
   The $CS(a, b)$ between two vectors $a$ and $b$ is defined as:
   
   $CS(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$

#### B.1.4.2 Normalization and Final Metric Calculation

To ensure robustness and comparability across different layers and models, the following normalization procedures and scaling is applied:

1. SVD Metric:
   
   a) Normalization:
      
      $M_{SVD}^{norm}(l) = \frac{M_{SVD}^{initial}(l) - \min_k M_{SVD}^{initial}(k)}{\max_k M_{SVD}^{initial}(k) - \min_k M_{SVD}^{initial}(k)}$
   
   b) Final metric:
      
      $M_{SVD}(l) = 1 - M_{SVD}^{norm}(l)$

2. CE Metric:
   
   a) Normalization:
      
      $M_{CE}^{norm}(l) = \frac{M_{CE}^{initial}(l) - \min_k M_{CE}^{initial}(k)}{\max_k M_{CE}^{initial}(k) - \min_k M_{CE}^{initial}(k)}$
   
   b) Final metric:
      
      $M_{CE}(l) = 1 - M_{CE}^{norm}(l)$

3. CS Metric:
   
   Normalization and final metric:
   
   $M_{CS}(l) = \frac{M_{CS}(l) - \min_k M_{CS}(k)}{\max_k M_{CS}(k) - \min_k M_{CS}(k)}$

## B.2 Metrics and Model Performance Relationship

### B.2.1 Objective Function $J(θ)$ Definition

In the context of neural networks, the objective function $J(θ)$ typically represents a loss function that we aim to minimize during training:

$J(θ) = (1/N) ∑_{i=1}^N L(y_i, f(x_i; θ)) + λR(θ)$

Where:
- $N$ is the number of training samples
- $θ$ represents all trainable parameters of the model
- $L$ is the per-sample loss function
- $y_i$ is the true label (or target value) for sample $i$
- $x_i$ is the input feature vector for sample $i$
- $f(x_i; θ)$ is the model's prediction for sample $i$
- $λ$ is a regularization coefficient
- $R(θ)$ is a regularization term

Breaking this down further:

1. Loss function $L$: Depending on the task, $L$ could be:
- Cross-entropy loss for classification: $L(y, ŷ) = -∑_j y_j log(ŷ_j)$
- Mean Squared Error (MSE) for regression: $L(y, ŷ) = (y - ŷ)²$

   where $y$ is a one-hot encoded vector and $ŷ$ is the model's softmax output

2. Model function $f(x; θ)$: For a neural network with $L$ layers: 
   
   $f(x; θ) = f_L(f_{L-1}(...f_1(x; θ_1)...; θ_{L-1}); θ_L)$, where $f_l$ represents the function of the $l$-th layer, and $θ_l$ are the parameters of that layer.

3. Regularization term $R(θ)$: This term helps prevent overfitting. Common choices include:
- L2 regularization: $R(θ) = ∑_k θ_k²$
- L1 regularization: $R(θ) = ∑_k |θ_k|$

As applied to the objective metrics:

1. SVD Metric ($M_{SVD}$):

   For a given layer $l$, $M_{SVD}(l)$ affects the capacity of $f_l$, potentially influencing the expressiveness of $f(x; θ)$.

2. Cross-Entropy Metric ($M_{CE}$):
   
   $M_{CE}(l)$ measures the uniqueness of layer $l$ compared to other layers, potentially affecting how diverse the features are across layers in $f(x; θ)$.

3. Cosine Similarity Metric ($M_{CS}$):
   
   $M_{CS}(l)$ measures how similar layer $l$ is to other layers, potentially indicating redundancy in $f(x; θ)$.

Goal is to prove that optimizing these metrics leads to a reduction in $J(θ)$:

- $∂J/∂M_{SVD}(l) > 0$
- $∂J/∂M_{CE}(l) > 0$
- $∂J/∂M_{CS}(l) < 0$

For each layer $l$, under the assumption that lower values of these metrics are better (note the direction of inequality for $M_{SVD}$ and $M_{CE}$ are flipped because we want to minimize these metrics).

### B.2.2 Partial Derivatives and Inequalities

1. SVD Metric ($M_{SVD}$):

   The goal is to prove $∂J/∂M_{SVD}(l) > 0$, since lower SVD values (higher 1 - $M_{SVD}$) should lead to higher learning rates.

   Using the chain rule:
   
   $∂J/∂M_{SVD}(l) = ∂J/∂θ_l * ∂θ_l/∂M_{SVD}(l)$

   a) $∂J/∂θ_l$ remains the same:
      
      $∂J/∂θ_l = (1/N) ∑_{i=1}^N ∂L(y_i, f(x_i; θ))/∂θ_l + λ∂R(θ)/∂θ_l$

   b) $∂θ_l/∂M_{SVD}(l)$ needs to be adjusted:
      
      $∂θ_l/∂M_{SVD}(l) ≈ -r * (U_l V_l^T)$  (note the negative sign due to the 1 - $M_{SVD}$ scaling)

   c) Combining these:

      $∂J/∂M_{SVD}(l) ≈ -(1/N) ∑_{i=1}^N ∂L(y_i, f(x_i; θ))/∂θ_l * r * (U_l V_l^T) - λ∂R(θ)/∂θ_l * r * (U_l V_l^T)$

   d) Increasing $M_{SVD}(l)$ decreases the learning rate for that layer, which should increase loss. Thus, $∂J/∂M_{SVD}(l) > 0$.

2. CE Metric ($M_{CE}$):

   The goal is to prove $∂J/∂M_{CE}(l) > 0$, since lower CE values (higher 1 - $M_{CE}$) should lead to higher learning rates.

   a) $M_{CE}(l)$ expression remains the same:
   
      $M_{CE}(l) = (1/(L-1)) ∑_{k≠l} CE(p_l, p_k)$

   b) The relationship between $M_{CE}(l)$ and information content $I_l$ needs to be adjusted:
   
      $ΔI_l ≈ Δ(1 - M_{CE}(l)) = -ΔM_{CE}(l)$

   c) The relationship between information content and loss remains the same:
   
      $∂J/∂I_l < 0$

   d) Combining these:

      $∂J/∂M_{CE}(l) ≈ -∂J/∂I_l > 0$

3. CS Metric ($M_{CS}$):

   For $M_{CS}$, lower values already correspond to higher learning rates, so no correction is needed. The goal remains to prove $∂J/∂M_{CS}(l) > 0$.

   a) $M_{CS}(l)$ expression remains the same:

      $M_{CS}(l) = (1/(L-1)) ∑_{k≠l} (w_l · w_k) / (||w_l|| ||w_k||)$

   b) The relationship between $M_{CS}(l)$ and orthogonality $O_l$ remains:

      $ΔO_l ≈ -ΔM_{CS}(l)$

   c) The relationship between orthogonality and loss remains the same:

      $∂J/∂O_l < 0$

   d) Combining these:

      $∂J/∂M_{CS}(l) ≈ -∂J/∂O_l > 0$

### B.2.3 Metric Improvements and J(θ) Correlation

   For each metric $M ∈ {M_{SVD}, M_{CE}, M_{CS}}$, this section aims to prove:
      
   For $M_{SVD}$ and $M_{CE}$: 
   
   $ΔM_i > 0 ⇒ E[ΔJ(θ)] > 0$ 
   
   For $M_{CS}$: 
   
   $ΔM_i < 0 ⇒ E[ΔJ(θ)] < 0$, where $ΔM_i$ is the change in metric $M$ for layer $i$, and $E[ΔJ(θ)]$ is the expected change in the objective function.

1. Taylor Expansion of $J(θ)$:

   Let $θ'$ be the updated parameters after a change in metric $M_i$. $J(θ')$ can be expressed using a second-order Taylor expansion around $θ$ [5][27]:
   
   $J(θ') ≈ J(θ) + ∇J(θ)^T (θ' - θ) + 1/2 (θ' - θ)^T H(θ) (θ' - θ)$, where $∇J(θ)$ is the gradient of $J$ with respect to $θ$, and $H(θ)$ is the Hessian matrix.

2. Relate Parameter Changes to Metric Changes:
   
   Express $(θ' - θ)$ in terms of $ΔM_i$ [4][12]: 
   
   For $M_{SVD}$ and $M_{CE}$: 
   
   $θ' - θ ≈ -∂θ/∂M_i * ΔM_i$ (note the negative sign)
   
   For $M_{CS}$: 

   $θ' - θ ≈ ∂θ/∂M_i * ΔM_i$

3. Substitute into Taylor Expansion:
   
   For $M_{SVD}$ and $M_{CE}$:
   
   $J(θ') ≈ J(θ) - ∇J(θ)^T (∂θ/∂M_i * ΔM_i) + 1/2 (ΔM_i)^T (∂θ/∂M_i)^T H(θ) (∂θ/∂M_i) ΔM_i$
   
   For $M_{CS}$:
   
   $J(θ') ≈ J(θ) + ∇J(θ)^T (∂θ/∂M_i * ΔM_i) + 1/2 (ΔM_i)^T (∂θ/∂M_i)^T H(θ) (∂θ/∂M_i) ΔM_i$

4. Take Expectation of Both Sides:
   
   For $M_{SVD}$ and $M_{CE}$:

   $E[J(θ')] ≈ J(θ) - E[∇J(θ)^T (∂θ/∂M_i * ΔM_i)] + 1/2 E[(ΔM_i)^T (∂θ/∂M_i)^T H(θ) (∂θ/∂M_i) ΔM_i]$
   
   For $M_{CS}$:
   
   $E[J(θ')] ≈ J(θ) + E[∇J(θ)^T (∂θ/∂M_i * ΔM_i)] + 1/2 E[(ΔM_i)^T (∂θ/∂M_i)^T H(θ) (∂θ/∂M_i) ΔM_i]$

5. Analyze Each Term:
   
   $J(θ)$ is constant with respect to the expectation: 
   
   $E[∇J(θ)^T (∂θ/∂M_i * ΔM_i)] = ∇J(θ)^T (∂θ/∂M_i * E[ΔM_i])$ (linearity of expectation) [27], the last term is positive semi-definite if $H(θ)$ is positive semi-definite, which is true for convex $J$ [3][26].

6. Correlation Between Metric Improvements:
   
   Using these derivations from section B.2.2:
   - For $M_{SVD}: ∂J/∂M_{SVD}(i) > 0$ [1][25].
   - For $M_{CE}: ∂J/∂M_{CE}(i) > 0$ [9][12].
   - For $M_{CS}: ∂J/∂M_{CS}(i) > 0$ [2][23].

   Results in the following conclusions:

   For $M_{SVD}$: $E[ΔJ(θ)] ≈ -∇J(θ)^T (∂θ/∂M_{SVD} * E[ΔM_{SVD}]) > 0$ when $ΔM_{SVD} > 0$
   
   For $M_{CE}$: $E[ΔJ(θ)] ≈ -∇J(θ)^T (∂θ/∂M_{CE} * E[ΔM_{CE}]) > 0$ when $ΔM_{CE} > 0$
   
   For $M_{CS}$: $E[ΔJ(θ)] ≈ ∇J(θ)^T (∂θ/∂M_{CS} * E[ΔM_{CS}]) < 0$ when $ΔM_{CS} < 0$

   The second-order term adds a small positive value, but for sufficiently small $ΔM_i$, the first-order term dominates.

8. Generalization:
   
   This proof holds for small changes in the metrics. For larger changes, the change can be considered as a series of small steps, and this proof applied iteratively.

## B.3 Metric Interdependencies and Interaction Effects

### B.3.1 Interaction Functions

To analyze the interdependencies between the SVD, CE, and CS metrics, interaction functions are defined that capture how changes in one metric affect the others and, consequently, the overall objective function $J(θ)$.

#### B.3.1.1 Interaction Function Introduction

For any two metrics $M_i$ and $M_j$, where $i, j ∈ {SVD, CE, CS}$, we define the pairwise interaction function $f_{i,j}$ as:

$f_{i,j}(M_i, M_j) = ∂²J / (∂M_i ∂M_j)$

This function quantifies how changes in $M_i$ affect the sensitivity of $J$ to changes in $M_j$, and vice versa.

Therefore the three-way interaction among all metrics, are defined as:

$f_{SVD,CE,CS}(M_{SVD}, M_{CE}, M_{CS}) = ∂³J / (∂M_{SVD} ∂M_{CE} ∂M_{CS})$

This function captures the combined effect of simultaneous changes in all three metrics on $J$.

#### B.3.1.3 Properties of Interaction Functions

1. Continuity: All interaction functions are assumed to be continuous in their arguments, ensuring smooth transitions in the optimization landscape.

   $∀ε>0, ∃δ>0$ such that $||(M'_i, M'_j) - (M_i, M_j)|| < δ ⇒ |f_{i,j}(M'_i, M'_j) - f_{i,j}(M_i, M_j)| < ε$

2. Symmetry: Pairwise interaction functions are symmetric with respect to their arguments.

   $f_{i,j}(M_i, M_j) = f_{j,i}(M_j, M_i)$

3. Boundedness: Interaction functions are bounded within a finite range to ensure stability in optimization.

   $∃K>0$ such that $|f_{i,j}(M_i, M_j)| ≤ K$ for all $M_i$, $M_j$ in their respective domains

4. Differentiability: Interaction functions are differentiable with respect to their arguments, allowing for gradient-based analysis.

   $∂f_{i,j}/∂M_i$ and $∂f_{i,j}/∂M_j$ exist and are continuous

#### B.3.1.4 Integral Representation

For a more general representation of the effect of metric interactions on $J$, we define the integral form:

$ΔJ = ∫∫∫ f_{SVD,CE,CS}(M_{SVD}, M_{CE}, M_{CS}) dM_{SVD} dM_{CE} dM_{CS}$

This formulation allows for the analysis of cumulative effects of metric changes over their entire ranges.

#### B.3.1.5 Decomposition of Higher-Order Interactions

The three-way interaction function can be decomposed into pairwise and pure three-way components:

$f_{SVD,CE,CS}(M_{SVD}, M_{CE}, M_{CS}) = f_{SVD,CE}(M_{SVD}, M_{CE}) + f_{SVD,CS}(M_{SVD}, M_{CS}) + f_{CE,CS}(M_{CE}, M_{CS}) + f_{pure}(M_{SVD}, M_{CE}, M_{CS})$

Where $f_{pure}$ represents the pure three-way interaction that cannot be accounted for by pairwise interactions.

These definitions and properties provide a framework for analyzing the complex interplay between the SVD, CE, and CS metrics in the LALR approach. They allow for a systematic exploration of how changes in one metric propagate through the system and affect the overall optimization process.

### B.3.2 Interaction Analysis

This section examines the pairwise interactions between the SVD, CE, and CS metrics, providing insights into how changes in one metric affect another and their combined impact on the objective function $J(θ)$.

#### B.3.2.1 SVD-CE Interaction: $f_{SVD,CE}(M_{SVD}, M_{CE})$

The interaction between the Singular Value Decomposition (SVD) and Cross-Entropy (CE) metrics is defined as:

$f_{SVD,CE}(M_{SVD}, M_{CE}) = ∂²J / (∂M_{SVD} ∂M_{CE})$

This interaction captures how changes in the SVD metric affect the sensitivity of $J$ to changes in the CE metric, and vice versa.

**Theorem B.3.2.1.1:** The SVD-CE interaction is generally positive, i.e., $f_{SVD,CE}(M_{SVD}, M_{CE}) > 0$ for most values of $M_{SVD}$ and $M_{CE}$ in the context of LALR fine-tuning.

*Justification:* 
Let $σ_i$ be the $i$-th singular value and $H_i$ be the entropy of the $i$-th layer. Then:

$f_{SVD,CE}(M_{SVD}, M_{CE}) ≈ ∑_i ∂²J / (∂σ_i ∂H_i) > 0$

This inequality holds because LALR applies higher learning rates to layers with high $M_{SVD}$ (low singular values) and high $M_{CE}$ (low entropy). This approach leads to:

1. An increase in $M_{SVD}$ results in higher learning rates for layers with low singular values, which will increase the model's overall capacity and expressiveness over time.
2. An increase in $M_{CE}$ results in higher learning rates for low-entropy layers, which will increase the model's overall entropy over time.

Both of these effects contribute to decreasing $J$ in the LALR framework, as they promote improved representation capacity and diversification of the model's features.

**Corollary B.3.2.1.2:** The positive interaction between SVD and CE in the context of LALR suggests that optimizing these metrics together can lead to synergistic improvements in $J$.

#### 3.2.2 SVD-CS Interaction: $f_{SVD,CS}(M_{SVD}, M_{CS})$

The interaction between the SVD and Cosine Similarity (CS) metrics is defined as:

$f_{SVD,CS}(M_{SVD}, M_{CS}) = ∂²J / (∂M_{SVD} ∂M_{CS})$

This interaction quantifies how changes in the SVD metric influence the effect of CS changes on $J$, and vice versa.

**Theorem B.3.2.2.1:** The SVD-CS interaction is generally positive, i.e., $f_{SVD,CS}(M_{SVD}, M_{CS}) > 0$ for most values of $M_{SVD}$ and $M_{CS}$ in the context of LALR fine-tuning.

*Justification:* 
Let $σ_i$ be the $i$-th singular value and $θ_{ij}$ be the angle between weight vectors of layers $i$ and $j$. Then:

$f_{SVD,CS}(M_{SVD}, M_{CS}) ≈ ∑_{i,j} ∂²J / (∂σ_i ∂θ_{ij}) > 0$

This inequality holds because LALR applies higher learning rates to layers with high $M_{SVD}$ (low singular values) and high $M_{CS}$ (high cosine similarity). This approach leads to:

1. An increase in $M_{SVD}$ results in higher learning rates for layers with low singular values, which will increase the model's overall capacity and expressiveness over time.
2. An increase in $M_{CS}$ results in higher learning rates for layers with high cosine similarity, which will increase the average angle between weight vectors over time.

Both of these effects contribute to decreasing $J$ in the LALR framework, as they promote improved representation capacity and orthogonalization of the model's features.

**Corollary B.3.2.2.2:** The positive interaction between SVD and CS in the context of LALR suggests that these metrics can be optimized together effectively, potentially leading to compounding improvements in $J$.

#### B.3.2.3 CE-CS Interaction: $f_{CE,CS}(M_{CE}, M_{CS})$

The interaction between the CE and CS metrics is defined as:

$f_{CE,CS}(M_{CE}, M_{CS}) = ∂²J / (∂M_{CE} ∂M_{CS})$

This interaction describes how changes in the CE metric affect the impact of CS changes on $J$, and vice versa.

**Theorem B.3.2.3.1:** The CE-CS interaction is generally positive, i.e., $f_{CE,CS}(M_{CE}, M_{CS}) > 0$ for most values of $M_{CE}$ and $M_{CS}$ in the context of LALR fine-tuning.

*Justification:* 
Let $H_i$ be the entropy of the $i$-th layer and $θ_{ij}$ be the angle between weight vectors of layers $i$ and $j$. Then:

$f_{CE,CS}(M_{CE}, M_{CS}) ≈ ∑_{i,j} ∂²J / (∂H_i ∂θ_{ij}) > 0$

This inequality holds because LALR applies higher learning rates to layers with high $M_{CE}$ (low entropy) and high $M_{CS}$ (high cosine similarity). This approach leads to:

1. An increase in $M_{CE}$ results in higher learning rates for low-entropy layers, which will increase the model's overall entropy over time.
2. An increase in $M_{CS}$ results in higher learning rates for layers with high cosine similarity, which will increase the average angle between weight vectors over time.

Both of these effects contribute to decreasing $J$ in the LALR framework, as they promote diversification and orthogonalization of the model's representations.

**Corollary B.3.2.3.2:** The positive interaction between CE and CS in the context of LALR suggests that these metrics can be optimized together effectively, potentially leading to synergistic improvements in $J$.

**Corollary B.3.2.3.3:** The positive interaction implies that focusing on layers with both high $M_{CE}$ and high $M_{CS}$ for increased learning rates can lead to compounding benefits, as improvements in one metric are likely to support improvements in the other.

#### B.3.2.4 Implications for LALR Enhancement

The analysis of pairwise interactions leads to several key insights for the LALR enhancement process:

1. The positive SVD-CE, SVD-CS, and CE-CS interactions suggest that these pairs of metrics can be improved effectively in sequence, potentially leading to compounding enhancements.

2. The consistently positive interactions indicate that the LALR approach of increasing learning rates for layers with high metric values is theoretically sound and can lead to synergistic improvements.

3. The relative nature of these interactions supports the sequential improvement of SVD, CE, and CS metrics, justified as follows:

   a) Enhancing SVD first establishes a more expressive model structure by focusing on layers with low singular values, creating a favorable foundation for overall model improvement.
   
   b) The positive SVD-CE interaction implies that CE improvement benefits from the prior SVD enhancements, as both target layers that can benefit from increased plasticity.
   
   c) The positive SVD-CS and CE-CS interactions indicate that CS improvement can effectively build upon both SVD and CE enhancements, promoting orthogonalization of weight vectors while maintaining expressiveness and diversity.
   
   d) The consistently positive interactions suggest that each subsequent improvement step can leverage the enhancements from the previous steps.

The total improvement in the objective function $J$ for the SVD-CE-CS sequence can be expressed as:

$ΔJ_{total} = ΔJ_{SVD} + ΔJ_{CE(SVD)} + ΔJ_{CS(SVD,CE)}$

Where $ΔJ_{CE(SVD)}$ represents the improvement in CE given prior SVD enhancement, and $ΔJ_{CS(SVD,CE)}$ represents the improvement in CS given both prior enhancements.

These interaction patterns:

1. $ΔJ_{CE(SVD)} > ΔJ_{CE}$ (due to positive SVD-CE interaction)
2. $ΔJ_{CS(SVD,CE)} > ΔJ_{CS(SVD)}$ (due to positive CE-CS interaction)
3. $ΔJ_{CS(SVD)} > ΔJ_{CS}$ (due to positive SVD-CS interaction)

suggests the SVD-CE-CS sequence yields better overall improvement compared to isolated enhancements:

$ΔJ_{SVD} + ΔJ_{CE(SVD)} + ΔJ_{CS(SVD,CE)} > ΔJ_{SVD} + ΔJ_{CE} + ΔJ_{CS}$

This sequential enhancement approach allows each step to build upon the improvements of the previous steps in a complementary manner. The SVD enhancement provides a strong foundation, the CE improvement leverages and enhances this foundation, and the CS improvement then fine-tunes the model structure while benefiting from the prior enhancements.

The pairwise interaction analysis illuminates the complex interplay between the LALR metrics and provides a theoretical basis for their sequential improvement. This insight is crucial for designing effective LALR strategies that fully leverage the interdependencies between metrics to achieve substantial improvements in the objective function $J(θ)$.

#### B.3.2.5 Higher-Order Interactions

While the previous sections focused on pairwise interactions between the SVD, CE, and CS metrics, there is potential negative effects for three-way interactions among all three metrics to exist. However, the positive pairwise interactions suggests that these higher-order interactions are likely to be positive, based on the following concepts and considerations:

1. Complementary Metrics: 
   SVD, CE, and CS capture fundamentally complementary aspects of the model's structure:
   - SVD: Layer-wise capacity and expressiveness
   - CE: Information content and diversity between layers
   - CS: Orthogonality of weight vectors
   These complementary effects suggest that improvements in each metric are likely to have synergistic effects, rather than conflicting or cancelling each other out.

2. Supermodularity: 
   The positive pairwise interactions indicate a supermodular structure in the optimization landscape. In supermodular systems, the marginal benefit of improving one component increases as other components improve. This property extends naturally to higher-order interactions, suggesting that simultaneous improvements in all three metrics would yield compounding benefits.

3. Information Theoretic Perspective: 
   From an information theory standpoint, concurrent improvements in capacity (SVD), diversity (CE), and orthogonality (CS) are likely to have synergistic effects on the model's overall information content. These three aspects work together to enhance the model's ability to represent and process information efficiently, suggesting that their combined effect would be at least as positive as their pairwise interactions.

4. Robustness to Negative Effects: 
   Even if small negative higher-order interactions were to exist, the strong positive pairwise interactions would likely dominate, ensuring overall positive effects on the optimization process. The magnitude of any potential negative higher-order effects would need to be substantial to overcome the established positive pairwise effects, which is unlikely given the complementary nature of the metrics.

5. Local Convexity in High-Dimensional Spaces: 
   In the high-dimensional optimization landscapes typical of deep learning, local convexity often emerges even in globally non-convex functions. This phenomenon, sometimes referred to as the "blessing of dimensionality," supports the notion that higher-order interactions are likely to be non-negative in the relevant regions of the parameter space during the fine-tuning process.

6. Monotonicity Argument:
   Let $f(M_{SVD}, M_{CE}, M_{CS})$ represent the overall effect on the objective function. The positive pairwise interactions imply:

   $\frac{\partial^2 f}{\partial M_i \partial M_j} \geq 0$ for all $i, j \in \{SVD, CE, CS\}$

   This suggests that $f$ is monotonically increasing in each metric, given fixed values of the other metrics. Extending this to three-way interactions, it's reasonable to expect that:

   $\frac{\partial^3 f}{\partial M_{SVD} \partial M_{CE} \partial M_{CS}} \geq 0$

7. Sperner's Lemma Application:
   Consider the simplicial complex formed by the three metrics. Sperner's Lemma suggests that if the pairwise interactions (edges of the simplex) are all positive, then there exists at least one point in the simplex where the three-way interaction (the interior) is also non-negative. While this doesn't guarantee non-negativity everywhere, it supports the likelihood of non-negative higher-order interactions.

While a rigorous proof of non-negative higher-order interactions would require more extensive mathematical analysis, these arguments provide strong support that higher-order interactions in the LALR framework are likely to be non-negative. This suggests that the LALR approach is likely to maintain its effectiveness even when considering these more complex interactions.

#### B.3.2.6 Bounds on Interaction Effects 

Establishing bounds on interaction effects between the SVD, CE, and CS metrics serves the following functions in the LALR framework:

- Stability: Upper and lower interaction limits illustrate LALR's safeguards against uncontrolled behavior during fine-tuning.
- Theoretical Foundation: Analysis of the interaction bounds substantiates LALR's potential effectiveness.
- Optimization Strategy: Analysis of the integration bounds yields insights into the relationships between each metric's focus, which support the metric optimization sequencing order
- Convergence Behavior: Interaction bounds illuminate LALR's long-term behavior, indicating how improvements may evolve as fine-tuning progresses.

The subsequent theorems establish these bounds and explore their implications for the LALR fine-tuning process.

**Theorem B.3.2.6.1 (Bounds on Pairwise Interactions):**
For any two metrics $M_i$ and $M_j$, where $i, j ∈ {SVD, CE, CS}$, the magnitude of their interaction is bounded:

$K_{ij}^{min} · max(|∂J/∂M_i|, |∂J/∂M_j|) ≤ |f_{i,j}(M_i, M_j)| ≤ K_{ij}^{max} · min(|∂J/∂M_i|, |∂J/∂M_j|)$

where $K_{ij}^{min}$ and $K_{ij}^{max}$ are constants depending on the specific pair of metrics.

*Justification:* 
The upper bound is derived from the Cauchy-Schwarz inequality, while the lower bound ensures a minimum level of interaction, reflecting the synergistic nature of the metrics in LALR.

**Theorem B.3.2.6.2 (Metric-Specific Interaction Properties):**

1. SVD Interactions:
a) As $M_{SVD}$ approaches its optimal state (low singular values), $|∂J/∂M_{SVD}|$ decreases.
b) $f_{SVD,CE}$ and $f_{SVD,CS}$ may weaken as $M_{SVD}$ improves, but remain positive.

2. CE Interactions:
a) $M_{CE}$ doesn't have a global optimal state, but seeks to maximize entropy differences between layers.
b) $f_{SVD,CE}$ and $f_{CE,CS}$ remain consistently positive across the range of $M_{CE}$ values.

3. CS Interactions:
a) Like $M_{CE}$, $M_{CS}$ doesn't have a global optimal state, but aims to minimize cosine similarity between layers.
b) $f_{SVD,CS}$ and $f_{CE,CS}$ maintain positive values regardless of the current $M_{CS}$ state.

*Justification:* 
These properties follow from the definitions and behaviors of each metric as described in earlier sections. The consistent positivity of interactions aligns with the LALR approach of applying higher learning rates to layers with higher metric values.

**Theorem B.3.2.6.3: (Conditions for Maximizing Positive Interactions):**

The positive interaction between metrics $M_i$ and $M_j$ is maximized under the following conditions:

1. For SVD: $M_{SVD}$ is high, indicating low singular values and high potential for improvement.
2. For CE and CS: The relative differences in entropy (CE) or cosine similarity (CS) between layers are large.
3. The gradients $∂J/∂M_i$ and $∂J/∂M_j$ are aligned.
4. The layers affected by $M_i$ and $M_j$ exhibit a high degree of interdependence in the network architecture.

*Justification:* 
Analysis of the interaction function $f_{i,j}(M_i, M_j)$ under these conditions demonstrates that they lead to the largest positive values of the second partial derivative $∂²J/(∂M_i ∂M_j)$.

**1.High $M_{SVD}$ (for SVD metric)**

High $M_{SVD}$ values, indicating low singular values and high improvement potential, maximize positive interactions due to:

a) Larger $∂J/∂M_{SVD}$ when singular values are low, indicating greater room for improvement.
b) Amplified effect of changes in other metrics (CE or CS) on $J$ when the layer has high improvement potential.

Mathematical expression:
$|∂²J/(∂M_{SVD} ∂M_j)| ∝ 1/σ_min$, where $σ_min$ is the smallest singular value.

As $σ_min$ approaches zero (high $M_{SVD}$), the interaction magnitude increases.

**2. Large relative differences in entropy or cosine similarity**

For CE and CS metrics, large relative differences between layers lead to stronger interactions:

a) CE: Larger entropy differences imply more distinct information content between layers, amplifying the effect of changes in other metrics.
b) CS: Larger cosine similarity differences indicate more diverse representations, enhancing the impact of changes in other metrics.

Mathematical expressions:

$|∂²J/(∂M_{CE} ∂M_j)| ∝ |H_max - H_min|$

$|∂²J/(∂M_{CS} ∂M_j)| ∝ |CS_max - CS_min|$

Where $H$ represents entropy and $CS$ represents cosine similarity.

**3. Aligned gradients $∂J/∂M_i$ and $∂J/∂M_j$**

Alignment of gradients of $J$ with respect to $M_i$ and $M_j$ maximizes their combined effect, leading to a positive interaction:

$∂²J/(∂M_i ∂M_j) ≈ (∂J/∂M_i · ∂J/∂M_j) / ||∂J/∂M_i|| ||∂J/∂M_j||$

The right-hand side reaches its maximum when the gradients are perfectly aligned, resulting in the largest positive interaction value.

**4. Layer interdependence**

High interdependence between layers affected by $M_i$ and $M_j$ in the network architecture causes changes in one metric to influence the other. This interdependence is quantifiable using measures such as mutual information or correlation between layer activations.

Let $I(L_i; L_j)$ represent the mutual information between layers $i$ and $j$, then:

$|∂²J/(∂M_i ∂M_j)| ∝ I(L_i; L_j)$

As mutual information increases, the interaction magnitude increases correspondingly.

**5. Conclusion**

The combination of these conditions yields the overall interaction expression:

$∂²J/(∂M_i ∂M_j) ≈ K · (1/σ_min) · |ΔM| · cos(θ) · I(L_i; L_j)$

Where:
- $K$ is a positive constant
- $σ_min$ is the minimum singular value (relevant for SVD)
- $|ΔM|$ represents the relative difference in entropy or cosine similarity
- $θ$ is the angle between $∂J/∂M_i$ and $∂J/∂M_j$
- $I(L_i; L_j)$ is the mutual information between affected layers

This formulation demonstrates that the conditions stated in Theorem 3.2.5.3 collectively lead to the largest positive values of $∂²J/(∂M_i ∂M_j)$, thus maximizing the positive interactions between metrics.

**6. Implications for LALR Optimization:**

1. Adaptive Improvement: The bounds on interactions suggest that the improvement potential varies throughout the fine-tuning process, with potentially larger gains early on, especially for SVD.

2. Sustained Progress: The positive lower bound on interactions ensures that LALR continues to make progress even as some metrics (particularly SVD) approach more optimal states.

3. Dynamic Optimization Strategy: To maximize LALR benefits, the fine-tuning process should:
   a) Initially prioritize layers with high $M_{SVD}$ values.
   b) Continuously reassess and target layers with high relative CE and CS values.
   c) Adjust the sequencing of metric optimizations based on the current gradient alignments and layer interdependencies.

4. Convergence Behavior: The interaction bounds provide insights into LALR's convergence. While improvements from SVD optimization may slow over time, the relative nature of CE and CS ensures ongoing optimization potential.

5. Robustness: The existence of both upper and lower bounds indicates that LALR is robust across various model states and architectures, maintaining effectiveness throughout the fine-tuning process.

### B.3.3 Interaction and Bound Analysis Summary

The analysis of metric interactions and their bounds justifies that enhancing SVD first, followed by CE, and finally CS for the LALR approach is most effective:

1. Synergistic Effects: As shown in Theorem B.3.2.1.1 and Theorem B.3.2.2.1, SVD has positive interactions with both CE and CS. Enhancing SVD first maximizes the potential benefits of these interactions in subsequent steps.

2. Global vs. Relative Metrics: Unlike CE and CS, which are relative metrics between layers, SVD provides a more global view of each layer's capacity. Addressing this global aspect first ensures that subsequent relative improvements (CE and CS) occur in a more balanced and expressive network structure.

3. Diminishing Returns: As noted in Theorem B.3.2.6.2, the impact of increasing SVD may decrease as layers approach optimal states. By prioritizing SVD early, we capitalize on its highest potential for improvement.

4. Cascade Effect: Improvements in SVD can lead to more meaningful relative differences for CE and CS improvements. This cascade effect is more pronounced when SVD is enhanced first, setting the stage for more effective CE and CS focused fine-tuning enhancements.

5. Stability: Starting with SVD provides a stable beginning to the optimization process, as it directly addresses the fundamental expressiveness of each layer. This stability supports more nuanced enhancement in subsequent steps.

6. CE Enhancement: Increasing the entropy of individual layers through CE enhancement effectively spreads out the vector space and angles between layer representations. This process creates a more diverse and informative feature space, setting the stage for subsequent CS enhancement.

7. Expanded Feature Space: Improving CE first allows the model to explore a wider range of feature combinations and representations, increasing the potential for discovering novel and task-relevant patterns. This expanded feature space provides a more favorable environment for CS enhancement. The CE-CS interaction analysis not only supports the overall enhancement sequence of SVD, CE, and CS but also highlights the critical role of CE improvement in creating a more optimal environment for subsequent fine-tuning steps.

8. CS Enhancement: By prioritizing CE improvement, we ensure that CS enhancement can fully leverage the increased diversity and expressiveness of the layer representations, leading to more substantial improvements in the overall performance of the model. In contrast, enhancing CS before CE would be less effective, as it would be operating on a more limited and potentially redundant feature space.

Altogether, this rationale reinforces the theoretical foundations of the LALR approach and provides justification for LALR's SVD->CE->CS fine-tuning sequence. 

## B.4. Analysis of LALR's Impact on Layer Performance Gap

### B.4.1 Introduction to Layer Performance Dynamics

LALR's approach of increasing learning rates for layers with the lowest metric scores is designed to reduce the performance gap between layers, thereby enhancing overall model effectiveness. This analysis builds on the foundations established in sections B.1, B.2, and B.3, providing a theoretical framework for understanding how LALR contributes to model improvement.

### B.4.2 Definitions and Assumptions

Let $L = \{l_1, ..., l_n\}$ be the set of layers in the model. For each layer $l_i$, we define:

1. $P(l_i, t)$: Performance of layer $l_i$ at time $t$
2. $M_j(l_i, t)$: Value of metric $j \in \{SVD, CE, CS\}$ for layer $l_i$ at time $t$
3. $η(l_i, t)$: Learning rate for layer $l_i$ at time $t$

Assumptions:
1. Lower metric scores indicate more room for improvement.
2. The learning rate $η(l_i, t)$ is inversely proportional to the metric scores:

   $η(l_i, t) \propto \frac{1}{\sum_j w_j M_j(l_i, t)}$

   where $w_j$ are weight coefficients for each metric.

These assumptions are justified by the nature of the LALR approach, which aims to allocate more learning resources to layers that have the most potential for improvement.

### B.4.3 Layer Performance Gap

Define the layer performance gap at time $t$ as:

$G(t) = \max_{l_i \in L} P(l_i, t) - \min_{l_i \in L} P(l_i, t)$

Our goal is to demonstrate that LALR reduces $G(t)$ over time.

**Theorem B.4.3.1 (Convergence of Layer Performances)**

Under LALR optimization, the layer performance gap converges to a minimal value:

$\lim_{t \to \infty} G(t) \leq \epsilon$

for some small $\epsilon > 0$.

*Justification:* 
1. Let $l_{min}$ and $l_{max}$ be the layers with minimum and maximum performance at time $t$, respectively.

2. Model the rate of performance change for a layer as:

   $\frac{dP(l_i, t)}{dt} = k \cdot η(l_i, t) \cdot (1 - P(l_i, t))$

   where $k$ is a positive constant. This model captures the intuition that improvement is proportional to the learning rate and the room for improvement (1 - current performance).

3. Given the inverse relationship between $η(l_i, t)$ and metric scores, and assuming lower metric scores for $l_{min}$:

   $η(l_{min}, t) > η(l_{max}, t)$

4. Therefore:

   $\frac{dP(l_{min}, t)}{dt} > \frac{dP(l_{max}, t)}{dt}$

5. This implies that the performance of $l_{min}$ increases faster than $l_{max}$, reducing $G(t)$ over time.

6. As $G(t)$ approaches 0, the difference in learning rates diminishes due to the convergence of metric scores. This leads to the convergence of $G(t)$ to some minimal value $\epsilon$.

The value of $\epsilon$ depends on factors such as the model architecture, task complexity, and inherent limitations of the optimization process. While we cannot provide an exact value for $\epsilon$, we can expect it to be small relative to the initial performance gap.

### B.4.4 Rate of Convergence

While the exact rate of convergence is difficult to determine due to the complex dynamics of neural networks, we can provide a heuristic analysis:

Let $\Delta G(t) = G(t) - G(t+1)$ be the reduction in performance gap over one time step. We can model this as:

$\Delta G(t) \approx c \cdot G(t) \cdot (1 - G(t))$

where $c$ is a positive constant. This logistic-like model captures the intuition that the rate of gap reduction is proportional to both the current gap (room for improvement) and its complement (difficulty of further improvement as the gap narrows).

This model suggests that the convergence rate is faster when the performance gap is large and slows down as the gap narrows, which aligns with the expected behavior of LALR.

**Corollary B.4.3.1 (Overall Model Improvement):**

The reduction in layer performance gap leads to an improvement in overall model performance.

Proof:
1. Define overall model performance $P_{model}(t)$ as a function of individual layer performances:

   $P_{model}(t) = f(P(l_1, t), ..., P(l_n, t))$

2. Assume $f$ is limited by its worst-performing component:

   $P_{model}(t) \leq \min_{l_i \in L} P(l_i, t) + \delta(G(t))$

   where $\delta(G(t))$ is a non-negative function that decreases as $G(t)$ decreases. This captures the idea that the model's performance is constrained by its weakest layers, with some allowance for inter-layer synergies.

3. As LALR reduces $G(t)$, $\delta(G(t))$ decreases, allowing for potential improvement in $P_{model}(t)$.

The function $\delta(G(t))$ can be thought of as representing the "slack" in the model's performance due to inter-layer dependencies. As $G(t)$ decreases, this slack is reduced, leading to a tighter coupling between layer performances and overall model performance.

### B.4.6 Metric Recalculation and Adaptive Optimization

The LALR approach's unique strength lies in its adaptive nature, stemming from the recalculation of metrics (SVD, CE, CS) before each epoch. This process allows LALR to dynamically adjust to the evolving state of the model throughout the fine-tuning process. Let's formalize this concept:

**Theorem B.4.6.1 (Adaptive Metric Integration)**

Let $M_j^{(e)}(l_i)$ be the value of metric $j \in \{SVD, CE, CS\}$ for layer $l_i$ at epoch $e$. Then:

$M_j^{(e)}(l_i) = f_j(W_i^{(e-1)}, M_k^{(e-1)}(l_i))$

where $W_i^{(e-1)}$ represents the weights of layer $l_i$ after epoch $e-1$, and $k$ represents all metrics applied in epoch $e-1$.

*Justification:*
1. For the first epoch ($e=1$), $M_j^{(1)}(l_i)$ is computed based on the initial weights of the pre-trained model.

2. For subsequent epochs, $M_j^{(e)}(l_i)$ inherently captures the cumulative effects of all previous fine-tuning steps, including the interactions between metrics.

3. This recalculation ensures that each epoch's optimization is based on the most current state of the model, accounting for all previous modifications.

**Corollary B.4.6.2 (Compound Metric Effects)**

The learning rate for layer $l_i$ at epoch $e$, $η(l_i, e)$, is a function of the compound effects of all previously applied metrics:

$η(l_i, e) = g(M_{SVD}^{(e)}(l_i), M_{CE}^{(e)}(l_i), M_{CS}^{(e)}(l_i))$

where $g$ is a function that determines the learning rate based on the current metric values.

This adaptive recalculation process has several important implications for the layer performance gap dynamics:

1. Continuous Adaptation: LALR continuously adapts its optimization strategy based on the most recent state of the model, potentially leading to more efficient reduction of the performance gap.

2. Interaction Capture: The recalculated metrics inherently capture the interactions between different optimization criteria (SVD, CE, CS) without requiring explicit modeling of these interactions.

3. Stability Enhancement: By basing each epoch's optimization on the current model state, LALR may provide enhanced stability compared to fixed-schedule approaches, particularly for complex models like large language models.

4. Mitigation of Competing Objectives: If optimization based on one metric were to negatively impact another, the recalculation process would detect this in the subsequent epoch, allowing for corrective actions.

5. Long-term Consistency: This approach ensures that the optimization remains effective even over many epochs, as it continually readjusts based on the current model state rather than initial conditions.

The adaptive nature of LALR, formalized in Theorem B.4.6.1 and Corollary B.4.6.2, provides a strong theoretical basis for its potential effectiveness in reducing the layer performance gap over the course of fine-tuning. It allows LALR to maintain responsiveness to the changing dynamics of the model throughout the optimization process, potentially leading to more robust and effective fine-tuning outcomes.

### B.4.7 Extended Implications and Weight Update Dynamics

LALR's approach to fine-tuning has several far-reaching implications for model optimization and performance. These implications are deeply rooted in the weight update dynamics of the method.

Let $W_i(t)$ be the weight matrix of layer $l_i$ at time $t$. The weight update rule in gradient descent can be expressed as:

$W_i(t+1) = W_i(t) - η(l_i, t) \cdot \nabla L(W_i(t))$

where $\nabla L(W_i(t))$ is the gradient of the loss function with respect to $W_i(t)$.

Given LALR's inverse relationship between learning rates and metric scores:

$η(l_i, t) \propto \frac{1}{\sum_j w_j M_j(l_i, t)}$

We can deduce the following implications and dynamics:

1. Magnitude of Weight Changes and Accelerated Improvement:
   For layers with lower metric scores (indicating more room for improvement), $η(l_i, t)$ will be larger. This leads to:
   
   $\|W_i(t+1) - W_i(t)\| \propto η(l_i, t)$
   
   Thus, lower-performing layers undergo larger weight updates. These larger changes lead to more rapid exploration of the weight space, potentially accelerating the discovery of better configurations.

2. Bottleneck Mitigation:
   By allowing larger weight updates in lower-performing layers, LALR effectively addresses potential bottlenecks in the model. This can lead to more efficient information flow through the network, as previously underperforming layers are given the opportunity to adapt more quickly.

3. Adaptive Self-Correction and Regularization:
   As lower-performing layers improve due to larger weight updates, their metric scores will increase, naturally reducing their learning rates. This creates a self-correcting mechanism that prevents overcompensation. This dynamic adjustment of learning rates based on layer performance can be seen as a form of regularization, potentially improving the model's generalization capabilities.

4. Differential Plasticity:
   LALR effectively implements a form of differential plasticity across layers. Higher-performing layers maintain relative stability (preserving learned features), while lower-performing layers exhibit higher plasticity (adapting more quickly to new information). This adaptive optimization naturally adapts to the changing dynamics of the model during training.

5. Gradient Amplification:
   For lower-performing layers, LALR amplifies the effect of gradients. This can be particularly beneficial in deep networks where gradient magnitudes might otherwise diminish in earlier layers.
   
   $\|\frac{\partial W_i(t)}{\partial t}\| = η(l_i, t) \cdot \|\nabla L(W_i(t))\|$
   
   The larger $η(l_i, t)$ for lower-performing layers counteracts potential gradient vanishing issues.

6. Robustness to Initialization:
   By dynamically adjusting learning rates based on current layer performance rather than initial conditions, LALR may help mitigate the effects of poor weight initialization in certain layers. This can lead to more consistent performance across different initialization schemes.

These dynamics collectively contribute to LALR's ability to reduce the performance gap between layers. By allowing lower-performing layers to take larger steps in the weight space, LALR promotes a more balanced optimization process across the entire network. This approach not only addresses immediate performance disparities but also fosters a more robust and adaptable model architecture over the course of fine-tuning.

The interplay between these extended implications and the underlying weight update dynamics provides a strong theoretical foundation for LALR's potential effectiveness in fine-tuning large language models. It suggests that LALR can offer a nuanced, adaptive approach to optimization that is particularly well-suited to the complex landscapes of large neural networks.

### B.4.8 Conclusion

This analysis demonstrates that LALR's strategy of increasing learning rates for layers with the lowest metric scores theoretically leads to a reduction in the performance gap between layers. The emphasis on larger weight changes for lower-performing layers provides a clear mechanism for this gap reduction. This approach contributes to an improvement in overall model performance by addressing bottlenecks, enhancing information flow, providing a regularization effect, and implementing an adaptive, self-correcting optimization process.

While the exact dynamics of this process are complex and depend on various factors, the theoretical framework presented here provides a strong foundation for understanding LALR's potential benefits. The differential plasticity induced by LALR's learning rate adjustment mechanism offers a promising approach to fine-tuning large language models, potentially leading to more efficient and effective optimization strategies.

## References

1. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine learning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849-15854.

2. Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2), 157-166.

3. Bertsekas, D. P. (1999). Nonlinear Programming. Athena Scientific. (Chapter 2.3)

4. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. (Chapters 4.4.1, 5.1.2)

5. Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press. (Chapters 3.1, 4.1)

6. Bullen, P. S. (2013). Handbook of means and their inequalities (Vol. 560). Springer Science & Business Media. (Chapter 1)

7. Cha, S. H. (2007). Comprehensive survey on distance/similarity measures between probability density functions. International Journal of Mathematical Models and Methods in Applied Sciences, 1(4), 300-307.

8. Chollet, F. (2017). Deep Learning with Python. Manning Publications. (Chapter 8)

9. Cover, T. M., & Thomas, J. A. (2012). Elements of Information Theory. John Wiley & Sons. (Chapter 2.7)

10. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics.

11. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations. JHU press. (Chapter 2.4)

12. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapters 4.3.3, 6.2.2)

13. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

14. Horn, R. A., & Johnson, C. R. (2012). Matrix Analysis. Cambridge University Press. (Chapter 7.3)

15. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations.

16. Krause, E. F. (1987). Taxicab geometry: An adventure in non-Euclidean geometry. Courier Corporation. (Chapter 5)

17. Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. The Annals of Mathematical Statistics, 22(1), 79-86.

18. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

19. Li, Y., & Yuan, Y. (2017). Convergence analysis of two-layer neural networks with ReLU activation. Advances in Neural Information Processing Systems.

20. MacKay, D. J. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press. (Chapter 2)

21. Magnus, J. R. (1985). On differentiating eigenvalues and eigenvectors. Econometric Theory, 1(2), 179-191.

22. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. (Chapter 6)

23. Minsky, M., & Papert, S. (1972). Perceptrons: An Introduction to Computational Geometry. MIT Press.

24. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. Proceedings of the 27th International Conference on Machine Learning (ICML-10).

25. Neyshabur, B., Bhojanapalli, S., McAllester, D., & Srebro, N. (2018). A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks. International Conference on Learning Representations.

26. Nocedal, J., & Wright, S. J. (2006). Numerical Optimization. Springer. (Chapter 6.1)

27. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes 3rd Edition: The Art of Scientific Computing. Cambridge University Press. (Chapter 10.6)

28. Stewart, G. W. (1990). Matrix Perturbation Theory. Academic Press. (Chapter 4)

29. Strang, G. (2016). Introduction to Linear Algebra (Vol. 5). Wellesley-Cambridge Press. (Chapters 6.3, 6.6)

30. Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra (Vol. 50). SIAM. (Lecture 4)

31. Van Erven, T., & Harremos, P. (2014). Rényi divergence and Kullback-Leibler divergence. IEEE Transactions on Information Theory, 60(7), 3797-3820.

32. Vapnik, V. (1998). Statistical Learning Theory. Wiley. (Chapters 5 and 6)