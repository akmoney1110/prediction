import pandas as pd
import os
from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class Command(BaseCommand):
    help = "Train football prediction model from merged dataset using full feature set"

    def handle(self, *args, **kwargs):
        # Load dataset
        file_path = os.path.join("csv_data", "merged_full_football_data.csv")
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f"❌ File not found: {file_path}"))
            return

        df = pd.read_csv(file_path)

        # Build target from FTR (H = Home Win)
        if 'FTR' not in df.columns:
            self.stdout.write(self.style.ERROR("❌ 'FTR' column missing for target creation"))
            return
        df['target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)

        # Drop columns that leak match outcome or are irrelevant
        leak_cols = [
            'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
            'SourceFile', 'HomeWin', 'Date', 'Time'  # Date & Time may not be useful directly
        ]
        df = df.drop(columns=[col for col in leak_cols if col in df.columns])

        # Drop rows with missing values
        df = df.dropna()

        # One-hot encode categorical columns
        categorical_cols = ['Div', 'HomeTeam', 'AwayTeam']
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

        # Split into features and target
        X = df.drop(columns=['target'])
        y = df['target']

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.stdout.write(self.style.SUCCESS(f"✅ Model trained. Accuracy: {acc:.2%}"))

        # Save model
        model_path = os.path.join("predictor", "ml", "football_predictor.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        self.stdout.write(self.style.SUCCESS(f"✅ Model saved to {model_path}"))
