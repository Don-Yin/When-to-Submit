#!/Users/donyin/miniconda3/envs/common/bin/python

"""
1. (Ausloos, Nedič, Dekanski 2019) "Correlations between submission and acceptance of papers in peer review journals"
   Conclusions:
   - Seasonal bias in submission/acceptance rates.
   - Specialized vs interdisciplinary journals differ in seasonal patterns.
   - Some months (e.g., Jan/Feb) have higher acceptance probability for specialized journals.
   - Distinguishable peaks and dips observed seasonally.

2. (Ausloos, Nedič, Dekanski 2019) "Seasonal Entropy, Diversity and Inequality Measures of Submitted and Accepted Papers Distributions in Peer-Reviewed Journals"
   Conclusions:
   - Seasonal patterns can be described with entropy/diversity indices.
   - Entropy and diversity measures help distinguish features and evolutions in peer review processes.
   - Incorporating entropy/diversity as features can reflect non-uniform submission distributions.

3. (Putman et al. 2022) "Any Given Monday: Association Between Desk Rejections and Weekend Manuscript Submissions to Rheumatology Journals"
   Conclusions:
   - Weekend submissions more likely desk rejections.
   - Introduce a penalty for weekend submissions.

4. (Ausloos et al. 2017) "Day of the week effect in paper submission/acceptance/rejection to/in/by peer review journals. II. An ARCH econometric-like modeling"
   Conclusions:
   - Temporal autocorrelation and heteroskedasticity-like (ARCH) effects in submission/acceptance patterns.
   - Include lagged submission intensity or related features.

5. (Boja et al. 2018) "Day of the week submission effect for accepted papers in Physica A, PLOS ONE, Nature and Cell"
   Conclusions:
   - Specific day-of-week patterns correlate with acceptance likelihood.
   - Incorporate day-of-week as a predictive feature.

6. (Meng et al. 2020) "Getting a head start: turn-of-the-month submission effect for accepted papers in management journals"
   Conclusions:
   - Turn-of-the-month (ToM) effect: surge on 1st day of month for accepted papers, especially in management fields and weekends.
   - Add turn_of_month feature and small bonuses where appropriate.

7. (Shalvi et al. 2010) "Write when hot — submit when not: seasonal bias in peer review or acceptance?"
   Conclusions:
   - At a top psychology journal, summer peak in submissions but not in acceptance.
   - Submitting in summer may lower acceptance odds in "psych-like" scenario (field_type=0).
   - Add summer penalty if field=0.

8. (Schreiber 2012) "Seasonal bias in editorial decisions for a physics journal: you should write when you like, but submit in July"
   Conclusions:
   - In physics-like fields (field_type=1), July has highest acceptance rate.
   - Add July bonus if field=1.

9. (Sweedler 2020) "Strange Advice for Authors: Submit Your Manuscript with a Short Title on a Weekday"
   Conclusions:
   - Shorter titles and weekday submissions beneficial.
   - Add title_length feature: shorter title => bonus, longer title => penalty.
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

PAPER_CONCLUSIONS = {
    "Ausloos_2019_corr": "Seasonal bias differences.",
    "Ausloos_2019_ent": "Entropy/diversity for seasonal patterns.",
    "Putman_2022": "Weekend submissions => desk rejection penalty.",
    "Ausloos_2017_ARCH": "ARCH-like temporal autocorrelation.",
    "Boja_2018": "Day-of-week submission effect.",
    "Meng_2020": "Turn-of-month effect in management-like fields.",
    "Shalvi_2010": "Summer peak submissions but not acceptance for psych-like fields.",
    "Schreiber_2012": "July best acceptance month in physics-like fields.",
    "Sweedler_2020": "Short title & weekday submission beneficial.",
}


class SyntheticDataGenerator:
    def __init__(self, start_date="2015-01-01", end_date="2019-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    def generate(self):
        dates = pd.date_range(self.start_date, self.end_date, freq="D")
        df = pd.DataFrame({"date": dates})
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day
        # field_type: 0=psych-like (Shalvi), 1=physics-like (Schreiber)
        df["field_type"] = np.random.choice([0, 1], size=len(df))

        df["desk_rejection_policy"] = np.random.rand(len(df))
        df["seasonal_factor"] = np.sin((df["month"] - 1) * np.pi / 6)

        intensity = np.random.rand(len(df)) * 0.5
        for i in range(1, len(df)):
            intensity[i] += 0.5 * intensity[i - 1] * np.random.rand()
        lagged = np.roll(intensity, 1)
        lagged[0] = np.mean(lagged[1:])  # Fill first value with mean of rest
        df["lagged_intensity"] = lagged

        df["seasonal_diversity"] = 1 - np.abs(df["seasonal_factor"]) * 0.5
        df["turn_of_month"] = (df["day_of_month"] == 1).astype(int)
        df["title_length"] = np.random.randint(5, 50, size=len(df))

        base = 0.5

        # Day-of-week effects: Tues(1)/Thurs(3) penalty; weekend penalty
        dow_effect = np.zeros(len(df))
        dow_effect[df["day_of_week"] == 1] = -0.05
        dow_effect[df["day_of_week"] == 3] = -0.05
        weekend_effect = np.where(df["day_of_week"].isin([5, 6]), -0.1, 0)

        season_effect = df["seasonal_factor"] * 0.05
        specialised_bonus = np.where((df["field_type"] == 0) & (df["month"].isin([1, 2])), 0.05, 0)
        desk_reject_penalty = -0.05 * df["desk_rejection_policy"]
        intensity_effect = df["lagged_intensity"] * 0.02
        diversity_effect = (df["seasonal_diversity"] - 0.75) * 0.02

        # Shalvi_2010: if field=0 (psych), summer (6,7,8) penalty
        summer_penalty = np.where((df["field_type"] == 0) & (df["month"].isin([6, 7, 8])), -0.05, 0)

        # Schreiber_2012: if field=1 (physics), July bonus
        july_bonus = np.where((df["field_type"] == 1) & (df["month"] == 7), 0.05, 0)

        # Meng_2020: turn-of-month effect if field=0 (like management)
        tom_bonus = np.where((df["field_type"] == 0) & (df["turn_of_month"] == 1), 0.1, 0)
        weekend_tom_correction = np.where((df["field_type"] == 0) & (df["turn_of_month"] == 1) & (df["day_of_week"].isin([5, 6])), 0.05, 0)

        # Sweedler_2020: shorter title better
        avg_title_len = 27.5
        title_effect = np.where(
            df["title_length"] < avg_title_len, (avg_title_len - df["title_length"]) * 0.001, -(df["title_length"] - avg_title_len) * 0.001
        )

        df["score"] = (
            base
            + dow_effect
            + weekend_effect
            + season_effect
            + specialised_bonus
            + desk_reject_penalty
            + intensity_effect
            + diversity_effect
            + tom_bonus
            + summer_penalty
            + july_bonus
            + title_effect
            + weekend_tom_correction
        )

        df["score"] = df["score"].clip(0, 1)

        return df


class SubmissionScoreModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)

    def get_features(self):
        return [
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


def date_to_score(date_str, model):
    """
    date_str: YYYY-MM-DD
    """
    date = pd.to_datetime(date_str)
    test_df = pd.DataFrame({"date": [date]})
    test_df["day_of_week"] = test_df["date"].dt.dayofweek
    test_df["month"] = test_df["date"].dt.month
    test_df["day_of_month"] = test_df["date"].dt.day
    test_df["field_type"] = np.random.choice([0, 1], size=1)
    test_df["desk_rejection_policy"] = np.random.rand(1)
    test_df["seasonal_factor"] = np.sin((test_df["month"] - 1) * np.pi / 6)
    test_df["lagged_intensity"] = np.random.rand(1) * 0.5
    test_df["seasonal_diversity"] = 1 - np.abs(test_df["seasonal_factor"]) * 0.5
    test_df["turn_of_month"] = (test_df["day_of_month"] == 1).astype(int)
    test_df["title_length"] = np.random.randint(5, 50, size=1)

    features = model.get_features()
    pred = model.predict(test_df[features])
    return pred[0]


if __name__ == "__main__":
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

    preds = model.predict(X_val)
    mse = np.mean((preds - y_val) ** 2)
    print("Validation MSE:", mse)

    today = datetime.date.today()
    test_dates = pd.date_range(today, periods=7, freq="D")
    test_df = pd.DataFrame({"date": test_dates})
    test_df["day_of_week"] = test_df["date"].dt.dayofweek
    test_df["month"] = test_df["date"].dt.month
    test_df["day_of_month"] = test_df["date"].dt.day
    test_df["field_type"] = np.random.choice([0, 1], size=len(test_df))
    test_df["desk_rejection_policy"] = np.random.rand(len(test_df))
    test_df["seasonal_factor"] = np.sin((test_df["month"] - 1) * np.pi / 6)
    test_df["lagged_intensity"] = np.random.rand(len(test_df)) * 0.5
    test_df["seasonal_diversity"] = 1 - np.abs(test_df["seasonal_factor"]) * 0.5
    test_df["turn_of_month"] = (test_df["day_of_month"] == 1).astype(int)
    test_df["title_length"] = np.random.randint(5, 50, size=len(test_df))

    test_preds = model.predict(test_df[features])
    print("Predicted submission goodness scores for next 7 days:")
    for d, p in zip(test_df["date"], test_preds):
        print(d.strftime("%Y-%m-%d"), ":", p)

    sns.set_theme(
        style="darkgrid",
        rc={
            "axes.facecolor": "#2b2b2b",
            "figure.facecolor": "#1f1f1f",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "#404040",
        },
    )

    plt.figure(figsize=(7, 5))
    sns.lineplot(x=test_df["date"], y=test_preds, marker="o", color="#00ffff")
    plt.title("Predicted Submission Goodness Score (Next 7 Days)", color="white")
    plt.xlabel("Date", color="white")
    plt.ylabel("Predicted Score", color="white")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("submission_score.png", dpi=300, facecolor="#1f1f1f", edgecolor="none")

    # Example usage of the date_to_score function
    example_date = "2024-12-20"
    score_for_example = date_to_score(example_date, model)
    print(f"Score for {example_date}:", score_for_example)
