import pandas as pd


def slice_iris(df, feature):
    """Calculate descriptive stats on slices of the Iris dataset."""
    for cls in df["species"].unique():
        df_temp = df[df["species"] == cls]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {cls}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()


def main():
    df = pd.read_csv("iris.csv")

    slice_iris(df, "sepal_length")
    slice_iris(df, "sepal_width")
    slice_iris(df, "petal_length")
    slice_iris(df, "petal_width")


if __name__ == "__main__":
    main()
