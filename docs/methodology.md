# Methodology & Evaluation Protocol

This document provides the formal mathematical formulation, evaluation methodology, and anti-leakage principles for our capstone project.

---

## 1. Problem Formulation: The Ungauged Catchment Setting

Let $\mathcal{C} = \{c_1, \dots, c_N\}$ denote a regional collection of river catchments. For each catchment $c_i$:
- $\mathbf{X}_{\text{dyn}}^{(i)} \in \mathbb{R}^{T \times D_{\text{dyn}}}$ represents the daily meteorological and land surface forcing sequence over $T$ continuous days.
- $\mathbf{x}_{\text{static}}^{(i)} \in \mathbb{R}^{D_{\text{static}}}$ represents invariant catchment physiographic properties (topography, soil, geology, land use, dam storage).
- $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{E})$ represents the directed acyclic river network graph derived from HydroRIVERS, where nodes $v_i \in \mathcal{V}$ correspond to catchments, directed edges $(u, v) \in \mathcal{E}$ denote downhill river reaches, and $\mathbf{e}_{(u, v)}$ encodes routing distance and tributary area ratios.

### Operational Ungauged Constraint
For an unseen target catchment $c_{\text{target}} \in \mathcal{C}_{\text{test}}$, **historical observed streamflow $y_{\text{target}}(t)$ is strictly unavailable** during training, validation, and inference:
$$\hat{\mathbf{y}}_{\text{target}}(t) = f_{\boldsymbol{\theta}}\left( \mathbf{X}_{\text{dyn}}^{(\text{target})}(1:t), \, \mathbf{x}_{\text{static}}^{(\text{target})}, \, \mathcal{G}_{\text{target}} \right)$$
where $\hat{\mathbf{y}}_{\text{target}}(t) = \left[ \hat{q}_{\tau_1}(t), \hat{q}_{\tau_2}(t), \hat{q}_{\tau_3}(t) \right]$ represents the predicted discharge quantiles for $\tau \in \{0.10, 0.50, 0.90\}$.

---

## 2. Why Conventional Splitting Fails: Advective Network Leakage

In hydrological stream networks, surface water flows continuously down river channels:
$$Q_{\text{downstream}}(t) = Q_{\text{lateral}}(t) + \int Q_{\text{upstream}}(t - \Delta t) \, d\Delta t$$
If a model is evaluated using standard random catchment splitting:
1. An upstream tributary gauge $u$ may be assigned to the training set while a downstream trunk gauge $v$ is assigned to the test set.
2. Because downstream discharge is directly driven by routed upstream flow, the neural network learns to trivially interpolate streamflow between adjacent stations.
3. When evaluated, the model appears to perform exceptionally well, but when deployed on a genuinely ungauged river with no upstream/downstream gauges, it catastrophically fails.

### The Solution: River-Network-Aware Regional Basin Splits
Our framework partitions catchments at the **macro-basin boundary**, holding out entire river networks (**Mahanadi** and **Narmada**) for testing:
$$\mathcal{E}(\mathcal{V}_{\text{train}}, \mathcal{V}_{\text{test}}) = \emptyset, \quad \mathcal{E}(\mathcal{V}_{\text{val}}, \mathcal{V}_{\text{test}}) = \emptyset$$
This guarantees that **zero test catchment shares any upstream tributary, downstream mainstem, or shared drainage basin with any training catchment**.

---

## 3. Loss Formulation: Quantile Regression

To provide calibrated flood hazard bounds, the model optimizes the **Multi-Quantile Pinball Loss**:
$$\mathcal{L}_{\text{pinball}}(y, \hat{q}_\tau) = \max\left( \tau (y - \hat{q}_\tau), \, (\tau - 1)(y - \hat{q}_\tau) \right)$$
The total training objective across target quantiles $\tau \in \{0.10, 0.50, 0.90\}$ is:
$$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{C}_{\text{train}}|} \sum_{c \in \mathcal{C}_{\text{train}}} \frac{1}{\sum m_c(t)} \sum_{t=1}^T m_c(t) \sum_{\tau \in \{0.1, 0.5, 0.9\}} \mathcal{L}_{\text{pinball}}\left(y_c(t), \, \hat{q}_{\tau, c}(t)\right)$$
where $m_c(t) \in \{0, 1\}$ is a binary observation mask that excludes missing daily streamflow from the loss calculation.

---

## 4. Evaluation Metrics for Ungauged Benchmarking

Predicted streamflow on held-out test catchments is evaluated using standard hydrological performance metrics:

1. **Nash-Sutcliffe Efficiency (NSE)**:
   $$\text{NSE} = 1 - \frac{\sum_{t} (y(t) - \hat{q}_{0.5}(t))^2}{\sum_{t} (y(t) - \bar{y})^2}$$
2. **Kling-Gupta Efficiency (KGE)**:
   $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$
   where $r$ is the Pearson correlation, $\alpha = \sigma_{\hat{y}} / \sigma_y$ is the variability ratio, and $\beta = \mu_{\hat{y}} / \mu_y$ is the bias ratio.
3. **Peak Timing Error (PTE)**: Difference in days between observed and predicted annual maximum flood peaks.
4. **Prediction Interval Coverage Probability (PICP)**:
   $$\text{PICP} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left( \hat{q}_{0.1}(t) \le y(t) \le \hat{q}_{0.9}(t) \right) \quad (\text{Nominal target: } 80\%)$$
5. **Mean Prediction Interval Width (MPIW)**: Sharpness measure of interval uncertainty.
