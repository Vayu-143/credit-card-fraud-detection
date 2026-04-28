import numpy as np
import pandas as pd

def create_input_dataframe(amount, v1, v2, v3, v4, v5):
    random_features = np.random.normal(0, 1, 23)

    features = [0, v1, v2, v3, v4, v5] + list(random_features) + [amount]

    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    input_df = pd.DataFrame([features], columns=columns)

    return input_df