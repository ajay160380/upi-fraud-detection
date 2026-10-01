# Interview Notes: UPI Fraud Detection System

## 2-Minute Elevator Pitch
"I built an end-to-end UPI Fraud Detection system from scratch. The goal was to catch fraudulent transactions in real-time without relying on labeled fraud data, mirroring the reality of zero-day attacks. I generated 10,000 synthetic transactions spanning 30 days and injected 5 distinct fraud patterns. I engineered stateful features—like rolling averages, velocity counts, and cyclic hour encodings—strictly using historical data to avoid data leakage. For the model, I chose Isolation Forest because of its ability to isolate multidimensional anomalies. The model achieved a 70% F1-score. Finally, I deployed it as a FastAPI backend with a SQLite database to maintain transaction state, and built an interactive Streamlit dashboard that visualizes the risk scores, flags, and human-readable reasons in real-time. I also wrote parity tests ensuring zero skew between my batch pipeline and live API."

## Q&A

**1. Why use synthetic data instead of a real dataset like Kaggle's credit card fraud?**
*Answer:* I wanted to specifically model the nuances of India's UPI system—such as sender/receiver pairs, P2P vs P2M channels, and high-frequency, low-value transactions. Kaggle datasets are often heavily PCA-anonymized, which removes the opportunity to practice raw feature engineering like time-based grouping and entity behavior profiling.

**2. Why Isolation Forest over other Unsupervised methods (e.g. DBSCAN, Autoencoders)?**
*Answer:* Isolation Forest explicitly isolates anomalies rather than profiling normal data. It handles high-dimensional, mixed data types (numerical + categorical) well, scales linearly $O(N)$ with dataset size, and requires fewer hyperparameters than an Autoencoder, making it ideal for a fast, robust baseline in a production setting.

**3. Why did you set `contamination=0.05`?**
*Answer:* In real fraud detection, true fraud prevalence is usually < 1%. However, because I synthetically injected a 5% fraud rate to ensure enough positive samples for evaluation across all 5 fraud topologies, I aligned the model's expected contamination boundary to match the dataset's underlying reality. In production, this would be tuned heavily based on the business's tolerance for false positives.

**4. How did you avoid Data Leakage during feature engineering?**
*Answer:* In the batch pipeline (`features.py`), I calculated user-level historical stats using pandas `.shift()` combined with `.expanding()`, ensuring the current transaction was never included in its own historical mean/std. In the API (`app.py`), I explicitly queried the DB using `timestamp < current_transaction_timestamp` to recreate this strict barrier dynamically.

**5. How did you split your data, and why?**
*Answer:* I used a temporal split (e.g., first 21 days for training/validation, last 9 days for testing). Shuffling the data would have caused extreme future-leakage, as models would train on future states of a user. The validation split was used exclusively to grid-search threshold bounds and ensemble weights, leaving the test split totally unseen.

**6. What is Training-Serving Skew, and how does your Parity Test check it?**
*Answer:* Skew happens when the code generating features in production differs slightly from the code used in training (e.g., different missing-value defaults, floating point precision). My parity test takes 20 transactions from the CSV, passes them through the live FastAPI `/score` endpoint, and asserts that the resulting `risk_score` matches the batch-generated `features.csv` score with a tolerance of $<0.001$.

**7. Why didn't IQR and Time-Series boosting help, and what did you do about it?**
*Answer:* They were too noisy. They had very low precision (~12-16%) because benign users frequently deviate from their own historical bounds (e.g. buying a new laptop). During the validation grid search, I recognized this dragged down the ensemble's F1 score. Consequently, I dropped their weights to `0` for the final numerical score, but retained them as "explainers" to generate human-readable flags for the dashboard.

**8. Explain the Precision vs Recall trade-off in this project.**
*Answer:* Precision measures "how many flagged transactions were actually fraud?", while recall measures "how much of the total fraud did we catch?". A lower threshold yields high recall but low precision (many angry users blocked unnecessarily). A higher threshold yields high precision but low recall (fraud slips through, costing the bank money). The business must decide the optimal threshold.

**9. How would you handle Concept Drift?**
*Answer:* User behavior evolves. I would implement a continuous monitoring pipeline that tracks the distribution of incoming features (e.g., using Population Stability Index). If the distributions drift significantly from the training set, the system would trigger an automated retraining job on the most recent 30-60 days of data.

**10. How would you scale the API for production?**
*Answer:* SQLite locks on writes and is too slow for real-time reads under heavy concurrent load. I would migrate the transactional state to a distributed low-latency datastore like Redis or DynamoDB to fetch the user's historical aggregations instantly. The API itself would scale horizontally behind a load balancer.

**11. Why did you use cyclic encoding for the hour feature?**
*Answer:* If you just use the integer hour (0-23), the model thinks hour 23 and hour 0 are far apart (distance of 23), when in reality they are adjacent. Cyclic encoding uses sine and cosine transformations to map the hour onto a circle, explicitly teaching the model the continuous, cyclical nature of time. 

**12. What would you improve next?**
*Answer:* The model struggled with `velocity_burst` and `new_receiver_repeated` fraud. I would add a deterministic rules-engine layer on top of the ML model. If a user hits 10 transactions in 10 minutes, a strict rule should flag it, compensating for the Isolation Forest's weakness in detecting purely sequential anomalies.
