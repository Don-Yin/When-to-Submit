#!/Users/donyin/miniconda3/envs/common/bin/python
import numpy as np
import pandas as pd
import datetime
import json
from tqdm import tqdm
from main import date_to_score, SubmissionScoreModel, SyntheticDataGenerator
from sklearn.model_selection import train_test_split


def generate_year_scores():
    gen = SyntheticDataGenerator()
    df = gen.generate()

    features = [
        "day_of_week",
        "month",
        "day_of_month",
        "field_type",
        "desk_rejection_policy",
        "seasonal_factor",
        "lagged_intensity",
        "seasonal_diversity",
        "turn_of_month",
        "title_length",
    ]
    X = df[features]
    y = df["score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SubmissionScoreModel()
    model.fit(X_train, y_train)

    today = datetime.date.today()
    dates = pd.date_range(today, periods=365, freq="D")
    scores_dict = {}

    for date in tqdm(dates):
        date_str = date.strftime("%Y-%m-%d")
        scores = np.array([date_to_score(date_str, model) for _ in range(42)])
        avg_score = float(np.mean(scores))
        scores_dict[date_str] = avg_score

    # normalize scores between 0 and 1 (for visuals)
    min_score = min(scores_dict.values())
    max_score = max(scores_dict.values())
    scores_dict = {date: (score - min_score) / (max_score - min_score) for date, score in scores_dict.items()}

    with open("submission_scores.json", "w") as f:
        json.dump(scores_dict, f, indent=2)

    return scores_dict


if __name__ == "__main__":
    year_scores = generate_year_scores()
