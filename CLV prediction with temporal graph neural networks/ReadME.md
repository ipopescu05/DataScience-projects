# Modeling Relational Customer Dynamics with Temporal Graph Neural Networks

This repository contains the code and resources for the Master's Thesis: **"Embedding customer transactional data for lifetime value prediction: A comparative analysis of discrete and continuous temporal graph neural networks."**

---

## 1. The Big Idea: Beyond One-Off Models

In modern e-commerce, businesses face a significant challenge: understanding complex customer behavior from a constant stream of transactional data. The traditional approach involves building highly specialized, one-off models for each business question (e.g., churn prediction, product recommendation, CLV forecasting). This is inefficient, doesn't scale, and creates siloed views of the customer.

This project investigates a more modern and powerful paradigm: **learning a universal customer representation**.

The core idea is to distill all complex, relational transactional data into a single, rich numerical vector—an **embedding**—for each customer. This embedding acts as a form of "customer DNA," holistically capturing their habits, preferences, and position within the e-commerce ecosystem. Once learned, this universal embedding can be used as a simple, powerful input for numerous downstream tasks, dramatically simplifying the analytics pipeline.

To validate this approach, this research tests the quality of these learned embeddings on one of the most challenging downstream tasks: **Customer Lifetime Value (CLV) prediction**.

![Quintile Analysis R2 Plot](quantile_analysis_metrics.jpg)
*A key finding: The predictive power of all models is concentrated in the top 20% of customers (Quintile 5), where our proposed DTGNN model excels.*

---

## 2. Methodology: A Graph-Based Approach

The central hypothesis is that by representing customer-product interactions as a dynamic graph, we can capture relational patterns that traditional models miss.

### Models Compared:

1.  **BG/NBD + Gamma-Gamma (Benchmark):** The classic probabilistic model for CLV prediction. It uses only Recency, Frequency, and Monetary (RFM) data and treats each customer in isolation.

2.  **Discrete-Time TGNN (DTGNN) + LightGBM:**
    * **Representation:** The graph is viewed as monthly "snapshots."
    * **Key Feature:** This approach allows for rich, **dynamically changing node features**. For each snapshot, we engineer features like RFM metrics, lagged revenues, etc., for each customer.
    * **Architecture:** A HeteroGCLSTM model learns embeddings from the sequence of graph snapshots. These embeddings are then fed into a LightGBM model for the final CLV prediction.

3.  **Continuous-Time TGNN (CTGNN) + LightGBM:**
    * **Representation:** The graph is viewed as a continuous stream of timestamped events (purchases).
    * **Key Feature:** This approach relies on the model to learn historical patterns implicitly from the sequence of raw events, using only **static node features** (e.g., country).
    * **Architecture:** A Temporal Graph Network (TGN) model learns embeddings, which are then used by LightGBM.

---

## 3. Key Findings

This research yielded three primary conclusions:

1.  **Relational Learning is Superior:** The feature-rich **DTGNN model comprehensively outperforms the traditional BG/NBD benchmark**, achieving a 17.7% reduction in RMSE and a 45.4% reduction in sMAPE. This confirms that modeling the customer-product network captures significant predictive signal.

2.  **Feature Engineering is Still King:** The DTGNN, which was enhanced with thoughtfully engineered dynamic features, was significantly more effective than the more "end-to-end" CTGNN. This underscores that a hybrid approach combining domain knowledge with deep learning is highly effective.

3.  **Predictability is Concentrated:** The "Top 20% Phenomenon" was a critical discovery. For the bottom 80% of customers, no model had significant predictive power (R² ≈ 0). The success of all models was driven by their ability to predict the behavior of the highest-value customer segment. Our DTGNN model achieved the highest R² within this crucial segment.


## 4. For more information read the paper pdf file
