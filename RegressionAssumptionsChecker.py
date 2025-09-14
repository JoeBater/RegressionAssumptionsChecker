import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class RegressionAssumptionsChecker:
    """
    A diagnostic tool to check regression assumptions and recommend suitable algorithms.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Clean dataframe (no missing values, correctly typed).
    target : str
        Name of dependent variable column.
    algorithm : str, optional
        Name of algorithm to check assumptions for.
    """

    def __init__(self, df: pd.DataFrame, target: str, algorithm: str):
        self.df = df
        self.target = target
        self.algorithm = algorithm
        self.report = {}

    # ==================================================
    # 1. Core assumption checks
    # ==================================================
    def check_assumptions(self):
        """
        Run algorithm-specific assumption checks.
        Returns a structured dictionary with results.
        """
        results = {}

        if self.algorithm is None:
            results["status"] = "no_algorithm_specified"
            results["notes"] = "Use suggest_algorithms() to see permissible models."
            return results

        if self.algorithm == "LinearRegression":
            # Example placeholder checks:
            # (real implementation: VIF for multicollinearity,
            #  Breusch-Pagan for heteroscedasticity, etc.)
            results["linearity"] = True
            results["multicollinearity"] = False
            results["homoscedasticity"] = False

            if not results["homoscedasticity"]:
                results["status"] = "unsuitable"
                results["reason"] = (
                    "LinearRegression is not suitable because residuals are heteroscedastic. "
                    "Consider GradientBoostingRegressor, RandomForestRegressor, or HuberRegressor."
                )
            else:
                results["status"] = "suitable"

        elif self.algorithm in ["Ridge", "Lasso", "ElasticNet"]:
            results["status"] = "suitable"
            results["notes"] = "Handles multicollinearity better than OLS. Scaling required."

        elif self.algorithm in ["SVR", "KNeighborsRegressor"]:
            results["status"] = "conditionally_permissible"
            results["notes"] = "Sensitive to feature scale. Scaling required."

        elif self.algorithm in [
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ]:
            results["status"] = "suitable"
            results["notes"] = "Tree-based models are robust to collinearity and heteroscedasticity."

        else:
            results["status"] = "unknown"
            results["notes"] = f"Algorithm {self.algorithm} not recognized."

        self.report["assumptions"] = results
        return results

    # ==================================================
    # 2. Scaling advisor
    # ==================================================
    def scaling_recommendations(self):
        """
        Advise on scaling requirements for the chosen algorithm.
        """
        advice = {}
        alg = self.algorithm

        if alg in ["KNeighborsRegressor", "SVR"]:
            advice["required"] = True
            advice["recommended_scalers"] = ["StandardScaler", "MinMaxScaler"]
            advice["notes"] = "Apply scaling after train-test split."

        elif alg in ["Ridge", "Lasso", "ElasticNet"]:
            advice["required"] = True
            advice["recommended_scalers"] = ["StandardScaler", "RobustScaler"]
            advice["notes"] = "Scaling avoids penalization bias from feature magnitudes."

        elif alg in ["LinearRegression"]:
            advice["required"] = False
            advice["notes"] = "Scaling not required, though it may help coefficient interpretability."

        elif alg in [
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ]:
            advice["required"] = False
            advice["notes"] = "Scaling irrelevant; tree splits are scale-invariant."

        else:
            advice["required"] = None
            advice["notes"] = "Algorithm not recognized for scaling guidance."

        self.report["scaling"] = advice
        return advice

    # ==================================================
    # 3. Suggest permissible algorithms
    # ==================================================
    def suggest_algorithms(self):
        """
        Evaluate all supported algorithms and return a dict of permissible ones.
        """
        algorithms = [
            "LinearRegression", "Ridge", "Lasso", "ElasticNet",
            "SVR", "KNeighborsRegressor",
            "DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"
        ]

        suggestions = {}
        for alg in algorithms:
            self.algorithm = alg
            assumptions = self.check_assumptions()
            scaling = self.scaling_recommendations()

            if assumptions["status"] not in ["unsuitable", "unknown"]:
                suggestions[alg] = {
                    "status": assumptions["status"],
                    "notes": assumptions.get("notes", ""),
                    "scaling": scaling
                }

        self.report["suggestions"] = suggestions
        return suggestions
