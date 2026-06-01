import random


def predict(image_base64: str):

    scenario = random.choice([
        "low",
        "medium",
        "high"
    ])

    if scenario == "low":
        confidence = round(random.uniform(0.05, 0.29), 2)

    elif scenario == "medium":
        confidence = round(random.uniform(0.30, 0.69), 2)

    else:
        confidence = round(random.uniform(0.70, 0.99), 2)

    return {
        "insect_name": "Ant",
        "confidence": confidence
    }