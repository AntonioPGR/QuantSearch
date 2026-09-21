from typing import Literal
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

CorrelationType = Literal["pearsons", "spearmans", "kendalls"]

class CorrelationCalculator:
	"""Calculate correlation matrices between stock price series."""
	
	@staticmethod
	def calculate(closes: dict[str, pd.Series], correlation_type: CorrelationType = "pearsons", ) -> pd.DataFrame:
		"""Return a correlation matrix for the supplied closing prices."""
		
		if correlation_type not in CorrelationType:
			raise ValueError("correlation_type must be 'pearsons', 'spearmans', or 'kendalls'")
		
		stocks = list(closes.keys())
		
		correlations = pd.DataFrame(
			index=stocks,
			columns=stocks,
			dtype=float,
		)
		
		for stock in stocks:
			for stock2 in stocks:
				if stock == stock2:
					correlations.loc[stock, stock2] = 1.0
					continue
				
				x = closes[stock]
				y = closes[stock2]
				
				if correlation_type == "pearsons":
					value = CorrelationCalculator.calculate_pearsons(x, y)
				elif correlation_type == "spearmans":
					value = CorrelationCalculator.calculate_spearmans(x, y)
				else:
					value = CorrelationCalculator.calculate_kendalls(x, y)
				
				correlations.loc[stock, stock2] = value
		
		return correlations
	
	@staticmethod
	def calculate_pearsons(x, y) -> float:
		return round(pearsonr(x, y).statistic, 5)
	
	@staticmethod
	def calculate_spearmans(x, y) -> float:
		return round(spearmanr(x, y).statistic, 5)
	
	@staticmethod
	def calculate_kendalls(x, y) -> float:
		return round(kendalltau(x, y).statistic, 5)
