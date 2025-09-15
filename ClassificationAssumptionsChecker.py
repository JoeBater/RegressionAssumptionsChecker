class ClassificationAssumptionsChecker:
    def __init__(self, model=None):
        self.model = model
        self.assumption_results = {}

    def check_assumptions(self, X, y):
        self.assumption_results = {
            "multicollinearity": self._check_multicollinearity(X),
            "class_imbalance": self._check_class_imbalance(y),
            "scaling_issues": self._check_scaling(X),
            "separability": self._check_separability(X, y),
            "redundancy": self._check_redundancy(X),
        }
        return self.assumption_results

    def recommend_models(self):
        cautions = []
        if self.assumption_results.get("multicollinearity", False):
            cautions.append("Avoid LogisticRegression without regularization due to multicollinearity.")
        if self.assumption_results.get("class_imbalance", False):
            cautions.append("Consider models with class_weight support (e.g., RandomForest, SVC with weights).")
        if self.assumption_results.get("scaling_issues", False):
            cautions.append("Standardize features before using SVC, KNN, or LogisticRegression.")
        if self.assumption_results.get("separability", False):
            cautions.append("Linear models may struggle; consider tree-based or kernel methods.")
        return cautions