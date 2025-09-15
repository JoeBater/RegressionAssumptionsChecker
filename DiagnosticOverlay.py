class DiagnosticOverlay:
    def __init__(self, df, target, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.X = df.drop(columns=[target])
        self.y = df[target]

    # =======================================
    # Individual checks
    # =======================================
    def check_linearity(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import linear_reset

        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        reset_test = linear_reset(model, power=2, use_f=True)

        if self.visualize:
            import matplotlib.pyplot as plt
            sm.graphics.plot_partregress_grid(model)
            plt.suptitle("Partial Residuals")
            plt.show()

        return {
            "linearity": reset_test.pvalue >= 0.05,
            "p_value": reset_test.pvalue,
            "notes": "Linearity assumption holds." if reset_test.pvalue >= 0.05 else "Non-linearity detected."
        }

    def check_residual_normality(self):
        import statsmodels.api as sm
        from scipy.stats import shapiro
        import matplotlib.pyplot as plt

        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        stat, p_value = shapiro(model.resid)

        if self.visualize:
            sm.qqplot(model.resid, line='s')
            plt.title("Q-Q Plot of Residuals")
            plt.show()

        return {
            "normality": p_value >= 0.05,
            "p_value": p_value,
            "notes": "Residuals appear normal." if p_value >= 0.05 else "Residuals deviate from normality."
        }

    def check_scaling_need(self, threshold=10):
        stds = self.X.std()
        ratio = stds.max() / stds.min()

        return {
            "scaling_needed": ratio > threshold,
            "std_ratio": ratio,
            "notes": "Scaling recommended." if ratio > threshold else "Feature scales are aligned."
        }

    def check_multicollinearity(self, threshold=10):
        """Check multicollinearity using VIF, only if 2+ predictors.
        Returns diagnostics and minimal drop set recommendation.
        """
        X = self.df.drop(columns=[self.target])

        if X.shape[1] < 2:
            return {
                "multicollinearity": True,
                "details": None,
                "notes": "Only one predictor — multicollinearity check not applicable.",
                "recommendation": None
            }

        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import numpy as np
        import warnings
        import pandas as pd

        def compute_vif(data):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vifs = []
                for i in range(data.shape[1]):
                    try:
                        vif = variance_inflation_factor(data.values, i)
                    except np.linalg.LinAlgError:
                        vif = float("inf")  # or np.nan if you prefer
                    except ValueError:
                        vif = None
                    vifs.append(vif)
                return pd.DataFrame({
                    "feature": data.columns,
                    "VIF": vifs
                })

        vif_data = compute_vif(X)
        too_high = vif_data[vif_data["VIF"] > threshold]

        drop_set = []
        X_temp = X.copy()

        while not too_high.empty and X_temp.shape[1] > 1:
            # choose feature with highest VIF to drop
            to_drop = too_high.sort_values("VIF", ascending=False)["feature"].iloc[0]
            drop_set.append(to_drop)
            X_temp = X_temp.drop(columns=[to_drop])

            if X_temp.shape[1] == 0:
                break  # Prevent zero-size array crash

            vif_data = compute_vif(X_temp)
            too_high = vif_data[vif_data["VIF"] > threshold]

        if X_temp.shape[1] == 0:
            return {
                "multicollinearity": False,
                "details": [],
                "notes": "All predictors dropped due to perfect multicollinearity.",
                "recommendation": "Rebuild feature set to avoid linear dependence.",
                "drop_set": drop_set
            }
        
        recommendation = (
            f"High or perfect multicollinearity detected. "
            f"Consider removing these features: {drop_set}" if drop_set else None
        )
        return {
            "multicollinearity": len(drop_set) == 0,
            "details": vif_data.to_dict(orient="records"),
            "notes": (
                "Perfect or high multicollinearity detected" if drop_set 
                else "No severe multicollinearity."
            ),
            "recommendation": recommendation,
            "drop_set": drop_set
        }




    def check_heteroscedasticity(self):
        """Check heteroscedasticity using Breusch-Pagan test."""
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
        return {
            "homoscedasticity": lm_pvalue >= 0.05,
            "p_value": lm_pvalue,
            "notes": (
                "Homoscedasticity assumption holds." if lm_pvalue >= 0.05
                else "Residuals are heteroscedastic."
            )
        }

    def check_tree_suitability(self):
        X = self.df.drop(columns=[self.target])
        n_samples, n_features = X.shape

        issues = []

        if n_samples < 50:
            issues.append("Very small dataset — tree-based models may overfit.")
        if any(X[col].nunique() > n_samples * 0.5 for col in X.select_dtypes(include="object")):
            issues.append("High-cardinality categorical features may cause overfitting.")
        if n_features > n_samples:
            issues.append("More features than samples — trees may struggle.")

        return {
            "tree_suitability": len(issues) == 0,
            "notes": " ".join(issues) if issues else "No dataset characteristics preclude tree-based models."
        }
