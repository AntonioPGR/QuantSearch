import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

def calculateCorrelation(closes, type="pearsons"):
	correlations = pd.DataFrame(index=closes.keys(), columns=closes.keys(), dtype=float)
	for stock in closes.keys():
		for stock2 in closes.keys():
			if stock == stock2:
				correlations.loc[stock, stock2] = 1.0
			elif type == "pearsons":
				correlations.loc[stock, stock2] = calculatePearsons(closes[stock], closes[stock2])
			elif type == "spearmans":
				correlations.loc[stock, stock2] = calculateSpearmans(closes[stock], closes[stock2])
			else:
				correlations.loc[stock, stock2] = calculateKendalls(closes[stock], closes[stock2])
	return correlations

def calculatePearsons(x, y):
	return round(pearsonr(x, y).statistic, 5)

def calculateSpearmans(x, y):
	return round(spearmanr(x, y).statistic, 5)

def calculateKendalls(x, y):
	return round(kendalltau(x, y).statistic, 5)