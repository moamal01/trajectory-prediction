import pandas as pd
from pathlib import Path

HEIGHT = 720
WIDTH = 1280

RAW_ROOT = Path("../../data/raw")
NORM_ROOT = Path("../../data/normalized")

def normalize_and_save_data(category: str, n_games: int) -> None:
    for game_idx in range(1, n_games + 1):
        input_folder = RAW_ROOT / category / f"match{game_idx}" / "csv"
        output_folder = NORM_ROOT / category / f"match{game_idx}" / "csv"

        # Make directories
        csv_files = list(input_folder.glob("*.csv"))
        output_folder.mkdir(parents=True, exist_ok=True)

        # Normalize and save data
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            df["X"] = (df["X"] / WIDTH).round(3)
            df["Y"] = (df["Y"] / HEIGHT).round(3)

            df.to_csv(output_folder / csv_path.name, index=False)


# Run batches
normalize_and_save_data("Amateur", 3)
normalize_and_save_data("Professional", 23)
normalize_and_save_data("Test", 3)
