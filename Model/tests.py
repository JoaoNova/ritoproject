from sklearn.preprocessing import StandardScaler

x = [[112, 67, 32, 245, 22, 100, 1, 10, 56, 123], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
PredictorScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(x)

# Generating the standardized values of X and y
x = PredictorScalerFit.transform(x)

print(x)