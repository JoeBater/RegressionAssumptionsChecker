🧠 AssumptionsChecker Suite

Modular diagnostics for real-world ML workflows Built for deployment-aware pipelines, stakeholder clarity, and robust edge-case handling.

🔍 Overview

The AssumptionsChecker suite provides three modular tools — RegressionAssumptionsChecker and ClassificationAssumptionsChecker & DataIntegrityChecker — designed to surface hidden risks 
in machine learning models before they reach production. 

Whether you're validating a regression model’s residuals or stress-testing a classifier’s decision boundaries, these tools help you clarify, not just compute.

⚙️ Features

    ✅ Multicollinearity check via VIF with threshold flagging

    📈 Residual diagnostics: normality, skewness, kurtosis, Q-Q plots

    📊 Homoscedasticity tests: Breusch-Pagan, Goldfeld-Quandt

    🧠 Influence analysis: Cook’s distance, leverage scores

    📉 Classification diagnostics: class imbalance, confusion matrix, precision/recall drift

    🔍 Feature leakage detection via correlation and target leakage heuristics

    🧩 Modular overlays: plug-and-play diagnostics for any ML pipeline

    🗣️ Explanation-level control: toggle verbosity for technical vs stakeholder audiences

    📤 Export-ready reports: summary tables and visual diagnostics for review or presentation

🧠 Why This Matters

Most ML workflows skip assumption testing — until something breaks. This suite makes it easy to surface hidden issues early, communicate risks clearly, and build trust with stakeholders. It’s built for deployment, not just notebooks.

🛠️ Roadmap

    [ ] SHAP integration for residual impact

    [ ] Streamlit dashboard for stakeholder review

    [ ] CI/CD hooks for automated diagnostics in MLOps pipelines

    [ ] Time-series support for autocorrelation and drift detection
