# synthetic_data/ai_synthetic_data_generator.py
import logging
import numpy as np
import pandas as pd
from faker import Faker

logging.basicConfig(filename='logs/ai_synthetic_data_generator.log', level=logging.INFO)

faker = Faker()

# Generate synthetic dataset
def generate_synthetic_data(rows=1000):
    data = {
        "name": [faker.name() for _ in range(rows)],
        "age": np.random.randint(18, 70, size=rows),
        "income": np.random.randint(30000, 150000, size=rows),
        "credit_score": np.random.randint(300, 850, size=rows)
    }
    df = pd.DataFrame(data)
    df.to_csv("logs/synthetic_data.csv", index=False)
    logging.info("Synthetic data generated successfully.")
    return df

synthetic_data = generate_synthetic_data()
print("AI Synthetic Data Generation Completed.")