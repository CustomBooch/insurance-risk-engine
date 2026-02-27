from data.synthetic_generator import generate_synthetic_panel

df = generate_synthetic_panel()
print(df.head())
print(df.describe())