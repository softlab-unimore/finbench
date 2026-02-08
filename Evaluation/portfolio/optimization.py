from typing import Union, Optional
from enum import Enum

import numpy as np
import pandas as pd
from pypfopt import HRPOpt, EfficientFrontier, CLA, risk_models, expected_returns, \
    objective_functions, plotting
from matplotlib import pyplot as plt


class OptimizationMethod(str, Enum):
    """Enumeration of available portfolio optimization methods."""
    HRP = "hierarchical_risk_parity"
    MVO = "mean_variance_optimization"
    CLA = "critical_line_algorithm"


class RiskMethod(str, Enum):
    """Enumeration of available risk model methods."""
    SAMPLE_COV = "sample_cov"
    SEMICOVARIANCE = "semicovariance"
    EXP_COV = "exp_cov"
    LEDOIT_WOLF = "ledoit_wolf"
    LEDOIT_WOLF_CONST_VAR = "ledoit_wolf_constant_variance"
    LEDOIT_WOLF_SINGLE_FACTOR = "ledoit_wolf_single_factor"
    LEDOIT_WOLF_CONST_CORR = "ledoit_wolf_constant_correlation"
    ORACLE_APPROX = "oracle_approximating"


class ReturnMethod(str, Enum):
    """Enumeration of available return estimation methods."""
    MEAN_HISTORICAL = "mean_historical_return"
    EMA_HISTORICAL = "ema_historical_return"
    CAPM = "capm_return"


class ObjectiveFunction(str, Enum):
    """Enumeration of available optimization objectives."""
    SHARPE = "sharpe"
    VOLATILITY = "volatility"
    KELLY = "kelly"
    DEVIATION_RISK_PARITY = "deviation_risk_parity"
    TARGET_RETURN = "target_return"
    TARGET_VOLATILITY = "target_volatility"


class PortfolioOptimizer:
    """A unified class for different portfolio optimization methods."""

    def __init__(
            self,
            returns: pd.DataFrame,
            optimization_method: Union[OptimizationMethod, str] = OptimizationMethod.MVO,
            risk_method: Union[RiskMethod, str] = RiskMethod.LEDOIT_WOLF,
            return_method: Union[ReturnMethod, str] = ReturnMethod.MEAN_HISTORICAL,
            objective: Union[ObjectiveFunction, str] = ObjectiveFunction.SHARPE,
            l2_constraint: Optional[float] = None,
            target_return: Optional[float] = None,
            target_volatility: Optional[float] = None,
            risk_free_rate: float = 0.0,
            verbose: bool = False
    ):
        """
        Initialize the portfolio optimizer with the specified parameters.

        Args:
            returns: DataFrame of historical asset returns
            optimization_method: Method to use for optimization (HRP, MVO, CLA)
            risk_method: Method for risk estimation
            return_method: Method for return estimation
            objective: Optimization objective
            l2_constraint: L2 regularization parameter (for MVO)
            target_return: Target portfolio return (annualized)
            target_volatility: Target portfolio volatility (annualized)
            risk_free_rate: Annualized risk-free rate
            verbose: Whether to print additional information and plots
        """
        self.returns = returns

        # Convert string parameters to enum values if necessary
        self.optimization_method = self._parse_enum(optimization_method, OptimizationMethod)
        self.risk_method = self._parse_enum(risk_method, RiskMethod)
        self.return_method = self._parse_enum(return_method, ReturnMethod)
        self.objective = self._parse_enum(objective, ObjectiveFunction)

        self.l2_constraint = l2_constraint
        self.target_return = target_return
        self.target_volatility = target_volatility
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose

        # Validate parameters based on chosen method and objective
        self._validate_parameters()

        # Calculate risk and return estimates
        self.cov_matrix = risk_models.risk_matrix(
            returns,
            method=self.risk_method.value,
            returns_data=True
        )

        self.expected_returns = expected_returns.return_model(
            returns,
            method=self.return_method.value,
            returns_data=True
        )

        # Store optimization results
        self.weights = None
        self.performance_metrics = None

    @staticmethod
    def _parse_enum(value, enum_class):
        """Convert a string to the corresponding enum value if necessary."""
        if isinstance(value, enum_class):
            return value
        if isinstance(value, str):
            try:
                return enum_class(value)
            except ValueError:
                for enum_val in enum_class:
                    if enum_val.name.lower() == value.lower():
                        return enum_val
        raise ValueError(f"Invalid value '{value}' for {enum_class.__name__}")

    def _validate_parameters(self):
        """Validate the parameter combinations."""
        # Check optimization method and objective compatibility
        if self.optimization_method == OptimizationMethod.HRP and self.objective != ObjectiveFunction.SHARPE:
            raise ValueError("HRP method only supports Sharpe objective")

        if self.optimization_method == OptimizationMethod.CLA and self.objective not in [
            ObjectiveFunction.SHARPE, ObjectiveFunction.VOLATILITY
        ]:
            raise ValueError("CLA method only supports Sharpe and Volatility objectives")

        # Check target parameters
        if self.objective == ObjectiveFunction.TARGET_RETURN and self.target_return is None:
            raise ValueError("Target return must be specified when objective is 'target_return'")

        if self.objective == ObjectiveFunction.TARGET_VOLATILITY and self.target_volatility is None:
            raise ValueError("Target volatility must be specified when objective is 'target_volatility'")

    def optimize(self) -> dict[str, float]:
        """
        Run the optimization process based on the initialized parameters.

        Returns:
            Dictionary mapping asset names to their optimized weights
        """
        # Display data visualizations if in verbose mode
        if self.verbose:
            self._plot_input_data()

        # Select and run the appropriate optimization method
        if self.optimization_method == OptimizationMethod.HRP:
            self.weights = self._run_hrp()
        elif self.optimization_method == OptimizationMethod.MVO:
            self.weights = self._run_mvo()
        elif self.optimization_method == OptimizationMethod.CLA:
            self.weights = self._run_cla()

        # Display results if in verbose mode
        if self.verbose:
            self._plot_results()

        return self.weights

    def _run_hrp(self) -> dict[str, float]:
        """Run the Hierarchical Risk Parity optimization."""
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        weights = hrp.clean_weights()

        if self.verbose:
            self.performance_metrics = hrp.portfolio_performance(verbose=True)
            _, ax = plt.subplots(figsize=(8, 4))
            plotting.plot_dendrogram(hrp, ax=ax)
            plt.title("Hierarchical Clustering Dendrogram")
            plt.show()

        return weights

    def _run_mvo(self) -> dict[str, float]:
        """Run the Mean-Variance Optimization."""
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)

        # Apply L2 regularization if specified
        if self.l2_constraint:
            ef.add_objective(objective_functions.L2_reg, gamma=self.l2_constraint)

        # Apply the selected objective
        if self.objective == ObjectiveFunction.SHARPE:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif self.objective == ObjectiveFunction.VOLATILITY:
            ef.min_volatility()
        elif self.objective == ObjectiveFunction.KELLY:
            ef.nonconvex_objective(
                _kelly_objective,
                objective_args=(ef.expected_returns, ef.cov_matrix)
            )
        elif self.objective == ObjectiveFunction.DEVIATION_RISK_PARITY:
            ef.nonconvex_objective(
                _deviation_risk_parity,
                objective_args=(ef.cov_matrix,)
            )
        elif self.objective == ObjectiveFunction.TARGET_RETURN:
            ef.efficient_return(target_return=self.target_return)
        elif self.objective == ObjectiveFunction.TARGET_VOLATILITY:
            ef.efficient_risk(target_volatility=self.target_volatility)

        weights = ef.clean_weights()

        if self.verbose:
            self.performance_metrics = ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate,
                verbose=True
            )

        return weights

    def _run_cla(self) -> dict[str, float]:
        """Run the Critical Line Algorithm optimization."""
        cla = CLA(self.expected_returns, self.cov_matrix)

        if self.objective == ObjectiveFunction.SHARPE:
            cla.max_sharpe()
        elif self.objective == ObjectiveFunction.VOLATILITY:
            cla.min_volatility()

        weights = cla.clean_weights()

        if self.verbose:
            self.performance_metrics = cla.portfolio_performance(verbose=True)

        return weights

    def _plot_input_data(self):
        """Plot the input data (covariance matrix and expected returns)."""
        # Plot covariance/correlation matrix
        plotting.plot_covariance(self.cov_matrix, plot_correlation=True)
        plt.title(f"Covariance Matrix ({self.risk_method.value})")
        plt.tight_layout()
        plt.show()

        # Plot expected returns
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Expected Returns ({self.return_method.value})")
        self.expected_returns.sort_values().plot.barh(ax=ax)
        plt.tight_layout()
        plt.show()

    def _plot_results(self):
        """Plot the optimization results."""
        # Plot weights distribution
        _, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Portfolio Weights')
        pd.Series(self.weights).sort_values().plot.barh(ax=ax)
        plt.tight_layout()
        plt.show()

        # If using MVO, plot efficient frontier with random portfolios
        if self.optimization_method == OptimizationMethod.MVO and self.performance_metrics:
            expected_return, annual_vol, sharpe = self.performance_metrics

            # Generate random portfolios
            n_samples = 10000
            n_assets = len(self.expected_returns)
            w = np.random.dirichlet(np.ones(n_assets), n_samples)
            rets = w.dot(self.expected_returns)
            stds = np.sqrt((w.T * (self.cov_matrix @ w.T)).sum(axis=0))
            sharpes = rets / stds

            # Create the efficient frontier plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ef_plot = EfficientFrontier(self.expected_returns, self.cov_matrix)
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

            # Highlight the chosen portfolio
            ax.scatter(annual_vol, expected_return, marker='*', s=100, c='r', label='Optimal Portfolio')

            # Add random portfolios
            scatter = ax.scatter(stds, rets, marker='.', alpha=0.5, c=sharpes, cmap='viridis_r')

            # Add color bar for Sharpe ratio
            cbar = plt.colorbar(scatter)
            cbar.set_label('Sharpe Ratio')

            # Set plot labels and title
            ax.set_title('Efficient Frontier with Random Portfolios')
            ax.set_xlabel('Annual Volatility')
            ax.set_ylabel('Annual Expected Return')
            ax.legend()
            plt.tight_layout()
            plt.show()


# Utility functions for specific optimization objectives
def _kelly_objective(w, e_returns, cov_matrix, k=1):
    """Kelly criterion objective function for portfolio optimization."""
    variance = np.dot(w.T, np.dot(cov_matrix, w))
    objective = variance * 0.5 * k - np.dot(w, e_returns)
    return objective


def _deviation_risk_parity(w, cov_matrix):
    """Deviation risk parity objective function for portfolio optimization."""
    risk_contribution = w * np.dot(cov_matrix, w)
    risk_contribution_matrix = risk_contribution - risk_contribution.reshape(-1, 1)
    return (risk_contribution_matrix ** 2).sum().sum()
