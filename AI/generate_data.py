import random
import pandas as pd

NUMBER_OF_RECORDS = 5000

data = []

for _ in range(NUMBER_OF_RECORDS):

    # Generate information for a pair of passengers
    pickup_distance = round(random.uniform(0.1, 5.0), 2)
    time_difference = random.randint(0, 30)
    route_overlap = round(random.uniform(0, 100), 2)
    detour = round(random.uniform(0.1, 4.0), 2)
    destination_compatibility = round(random.uniform(0, 1), 2)
    preference_compatible = random.randint(0, 1)

    # Calculate a matching score
    score = 0

    # Route overlap is the most important factor
    if route_overlap >= 70:
        score += 30
    elif route_overlap >= 50:
        score += 20
    elif route_overlap >= 30:
        score += 10

    # Lower detour is better
    if detour <= 0.5:
        score += 25
    elif detour <= 1.5:
        score += 15
    elif detour <= 2.5:
        score += 8

    # Nearby pickup points are better
    if pickup_distance <= 0.5:
        score += 20
    elif pickup_distance <= 1.5:
        score += 12
    elif pickup_distance <= 3:
        score += 5

    # Smaller time difference is better
    if time_difference <= 5:
        score += 15
    elif time_difference <= 10:
        score += 10
    elif time_difference <= 20:
        score += 5

    # Destination compatibility
    if destination_compatibility >= 0.7:
        score += 7
    elif destination_compatibility >= 0.5:
        score += 4

    # Preference is a soft factor
    if preference_compatible == 1:
        score += 3

    # 1 = Good Match, 0 = Bad Match
    if score >= 50:
        match = 1
    else:
        match = 0

    # Add this passenger pair to the dataset
    data.append([
        pickup_distance,
        time_difference,
        route_overlap,
        detour,
        destination_compatibility,
        preference_compatible,
        match
    ])


columns = [
    "pickup_distance_km",
    "time_difference_min",
    "route_overlap_percent",
    "detour_km",
    "destination_compatibility",
    "preference_compatible",
    "match"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("ai/dataset/rickshaw_matching_dataset.csv", index=False)

print("Dataset generated successfully!")
print("Number of records:", len(df))

print("\nFirst 5 records:")
print(df.head())

print("\nMatch distribution:")
print(df["match"].value_counts())