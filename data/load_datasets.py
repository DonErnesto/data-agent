import os
from sklearn.datasets import load_wine, load_diabetes, fetch_california_housing
import pandas as pd

for dataset_loader, name in zip([load_wine, load_diabetes, fetch_california_housing],
    ["wine", "diabetes", "california_housing"]):
    csv_file_path = f"{name}.csv"
    print(f"processing {name}")
    if not os.path.exists(csv_file_path):
        # Load the  dataset
        dataset = dataset_loader()

        # The data is in a NumPy array, and the feature names are separate
        X = dataset.data
        feature_names = dataset.feature_names


        # Create a pandas DataFrame
        # It's good practice to include the target variable as well for a complete dataset
        df = pd.DataFrame(X, columns=feature_names)
        try: 
            y = dataset.target
            target_names = dataset.target_names
            df['target'] = y
            try:
                # You can replace the integer targets with the actual names if preferred
                df['target'] = df['target'].apply(lambda x: target_names[x])
            except TypeError:
                pass
        except AttributeError:
            print("not adding target_names, don't exist.")

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)

        print(f" dataset {name} loaded and saved to {csv_file_path}")
    else:
        print(f"not downloading or writing {name}, it already exists.")
