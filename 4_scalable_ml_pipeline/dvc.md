## Exercise local remote

git init
dvc init
mkdir ../local_remote
dvc remote add -d localremote ../local_remote

------- exercise_func.py
import sys
import pandas as pd


def create_ids(id_count: str) -> None:
    """ Generate a list of IDs and save it as a csv."""
    ids = [i for i in range(int(id_count))]
    df = pd.DataFrame(ids)
    df.to_csv("./id.csv", index=False)


if __name__ == "__main__":
    create_ids(sys.argv[1])

python ./exercise_func.py 10
dvc add id.csv
git add .gitignore id.csv.dvc
git commit -m "Initial commit of tracked sample.csv"
dvc push


## Exercise remote
conda install -c conda-forge dvc-gdrive
dvc remote add driveremote gdrive://UNIQUE_IDENTIFIER
dvc push --remote driveremote
dvc remote default newremote
dvc push

## Experiment tracking
dvc exp run -n clean_data \
            -p param \
            -d clean.py -d data/data.csv \
            -o data/clean_data.csv \
            python clean.py data/data.csv

-n specifies the stage name (as defined in the dvc.yaml)
-p specifies the parameters (as defined in the params.yaml)
-d provides the required files for this stage to run
-o specifies the output directory when the stage is completed

## Pipeline
import yaml
from yaml import CLoader as Loader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)

X = np.loadtxt("X.csv")
y = np.loadtxt("y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

lr = LogisticRegression(C=params["C"])
lr.fit(X_train.reshape(-1, 1), y_train)

preds = lr.predict(X_test.reshape(-1, 1))
f1 = f1_score(y_test, preds)
print(f"F1 score: {f1:.4f}")


---------------------
dvc exp run -n prepare -d fake_data.csv -d prepare.py -o X.csv -o y.csv python ./prepare.py


dvc exp run -n train -d X.csv -d y.csv -d train.py -p C python ./train.py


## Track experiments
dvc exp run -n evaluate \
            -d validate.py -d model.pkl \
            -M validation.json \
            python validate.py model.pkl validation.json

dvc exp diff
dvc exp show