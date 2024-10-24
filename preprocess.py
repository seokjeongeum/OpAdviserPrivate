import csv
import json
from pathlib import Path
import pprint

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    with open("scripts/experiment/gen_knobs/mysql_all_197_32G.json") as f1:
        all = json.load(f1)
        l = []
        for file_path in Path("repo").iterdir():
            with open(file_path) as f:
                for o in json.load(f)["data"]:
                    c = o["configuration"]
                    for key in all:
                        if key not in c:
                            c[key] = all[key]["default"]
                            if all[key]["type"] == "enum":
                                c[key] = str(c[key])
                    l.append(c)
        d = {k: v for k, v in all.items() if v["type"] == "enum"}
        transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        categories=list(
                            map(
                                lambda x: x["enum_values"],
                                d.values(),
                            )
                        )
                    ),
                    list(d.keys()),
                ),
            ],
            remainder="passthrough",
        )
        transformed_data = transformer.fit_transform(pd.DataFrame(l))
        transformed_data = pd.DataFrame(
            transformed_data, columns=transformer.get_feature_names_out()
        )
        transformed_data.to_csv("transformed.csv")
        print(transformed_data.shape)
