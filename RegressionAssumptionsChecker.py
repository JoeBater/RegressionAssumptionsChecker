import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from DiagnosticOverlay import DiagnosticOverlay

class RegressionAssumptionsChecker:
    def __init__(self, df, target, algorithm=None, visualize=False):
        self.df = df
        self.target = target
        self.algorithm = algorithm
        self.overlay = DiagnosticOverlay(df, target, visualize)
        self.report = {}

    # =======================================
    # Orchestrator
    # =======================================
    def check_assumptions(self):
        """Run relevant checks depending on algorithm."""
        results = {}

        if self.algorithm in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
            results["linearity"] = self.overlay.check_linearity()
            results["multicollinearity"] = self.overlay.check_multicollinearity()
            results["heteroscedasticity"] = self.overlay.check_heteroscedasticity()

            # Evaluate hard constraints
            if not results["heteroscedasticity"]["homoscedasticity"]:
                results["status"] = "unsuitable"
                results["reason"] = (
                    f"{self.algorithm} is not suitable because residuals are heteroscedastic. "
                    "Consider robust alternatives such as HuberRegressor or RANSACRegressor."
                )
            else:
                results["status"] = "suitable"

        elif self.algorithm in ["SVR", "KNeighborsRegressor"]:
            results["status"] = "conditionally_permissible"
            results["reason"] = "Scaling required."

        elif self.algorithm in ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]:
            tree_check = self.overlay.check_tree_suitability()
            results["tree_suitability"] = tree_check

            if not tree_check["tree_suitability"]:
                results["status"] = "conditionally_unsuitable"
                results["reason"] = tree_check["notes"]
            else:
                results["status"] = "suitable"
                results["reason"] = "Tree-based models do not assume linearity or homoscedasticity."

        else:
            results["status"] = "unknown"
            results["reason"] = f"Algorithm {self.algorithm} not recognized."
            self.help_algorithms()

        self.report["assumptions"] = results
        return results


    def help_algorithms(self):
        print('RegressionAssumptionsChecker')
        print('Checks data assumptions (linearity, multicolinearity, heteroscedasticity) for the algorithms:')
        print('LinearRegression, Ridge, Lasso, ElasticNet')
        print('SVR, KNeighborsRegressor')
        print('DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor')

    def export_report(self, format="dict"):
        if format == "json":
            import json
            return json.dumps(self.report, indent=2)
        return self.report