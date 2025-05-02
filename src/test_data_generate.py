import pandas as pd
import numpy as np
import random
import os

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define fixed categories for categorical columns
policy_states = ['IN', 'OH', 'IL']
policy_csls = ['100/300', '250/500', '500/1000']
insured_sexes = ['Male', 'Female']
education_levels = ['High School', 'Associate', 'College', 'Masters', 'PhD']
occupations = ['craft-repair', 'sales', 'exec-managerial', 'tech-support', 'machine-op-inspct']
hobbies = ['reading', 'chess', 'football', 'video-games', 'exercise']
relationships = ['husband', 'wife', 'own-child', 'unmarried', 'other-relative']
incident_types = ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car']
collision_types = ['Rear Collision', 'Side Collision', 'Front Collision']
incident_severities = ['Minor Damage', 'Major Damage', 'Total Loss']
authorities_contacted = ['Police', 'Fire', 'Ambulance', 'None']
incident_states = ['IN', 'OH', 'IL']
incident_cities = ['Indianapolis', 'Cleveland', 'Chicago']
auto_makes = ['Toyota', 'Ford', 'BMW', 'Chevrolet', 'Honda']
evidence_options = ['Yes', 'No']

# Generate 30 rows of synthetic data
records = []
for _ in range(30):
    record = {
        "months_as_customer": np.random.randint(1, 200),
        "age": np.random.randint(18, 80),
        "policy_annual_premium": round(np.random.uniform(300.0, 2000.0), 2),
        "umbrella_limit": random.choice([0, 1000000, 2000000]),
        "capital-gains": np.random.randint(0, 10000),
        "capital-loss": np.random.randint(0, 5000),
        "total_claim_amount": round(np.random.uniform(1000.0, 25000.0), 1),
        "injury_claim": round(np.random.uniform(0.0, 10000.0), 1),
        "property_claim": round(np.random.uniform(0.0, 10000.0), 1),
        "vehicle_claim": round(np.random.uniform(0.0, 10000.0), 1),
        "policy_state": random.choice(policy_states),
        "policy_csl": random.choice(policy_csls),
        "insured_sex": random.choice(insured_sexes),
        "insured_education_level": random.choice(education_levels),
        "insured_occupation": random.choice(occupations),
        "insured_hobbies": random.choice(hobbies),
        "insured_relationship": random.choice(relationships),
        "incident_type": random.choice(incident_types),
        "collision_type": random.choice(collision_types),
        "incident_severity": random.choice(incident_severities),
        "authorities_contacted": random.choice(authorities_contacted),
        "incident_state": random.choice(incident_states),
        "incident_city": random.choice(incident_cities),
        "auto_make": random.choice(auto_makes),
        "auto_year": np.random.randint(1995, 2021),
        "number_of_vehicles_involved": random.choice([1, 2, 3]),
        "bodily_injuries": random.choice([0, 1, 2]),
        "witnesses": random.choice([0, 1, 2, 3]),
        "evidence_available": random.choice(evidence_options)
    }
    records.append(record)

# Create DataFrame
df_new_claims = pd.DataFrame(records)

# Save to CSV
output_path = "../data/raw_data/new_claim.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_new_claims.to_csv(output_path, index=False)

print(f"✅ Generated test data and saved to: {output_path}")
