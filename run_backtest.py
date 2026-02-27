from data.synthetic_generator import generate_synthetic_panel
from backtest import expanding_window_backtest

df = generate_synthetic_panel()

results = expanding_window_backtest(df)

print("\nBacktest Summary:")
print(results)

print("\nAverage GLM Deviance:", results["glm_deviance"].mean())
print("Average LGBM Deviance:", results["lgbm_deviance"].mean())