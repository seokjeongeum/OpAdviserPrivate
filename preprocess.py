import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def custom_minmax_scaler(X, feature_range=(-1, 1), data_min=0, data_max=1):
    scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
    min_adj = feature_range[0] - data_min * scale
    return X * scale + min_adj


class CustomFunctionTransformer(FunctionTransformer):
    def __init__(self, kw_args, func=None, feature_names=None):
        super().__init__(func=func, kw_args=kw_args)
        self.feature_names = feature_names

    def get_feature_names_out(self, input_features=None):
        # Use provided feature names or fall back to input feature names
        return self.feature_names if self.feature_names else input_features


if __name__ == "__main__":
    with open("scripts/experiment/gen_knobs/mysql_all_197_32G.json") as f1:
        all = json.load(f1)
        enum = {k: v for k, v in all.items() if v["type"] == "enum"}
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
        transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        categories=list(
                            map(
                                lambda x: x["enum_values"],
                                enum.values(),
                            )
                        )
                    ),
                    list(enum.keys()),
                ),
            ]
            + [
                (
                    k,
                    CustomFunctionTransformer(
                        kw_args={
                            "feature_range": (-1, 1),
                            "data_min": v["min"],
                            "data_max": v["max"],
                        },
                        func=custom_minmax_scaler,
                        feature_names=[k],
                    ),
                    [k],
                )
                for k, v in {
                    k: v for k, v in all.items() if v["type"] == "integer"
                }.items()
            ],
            remainder="passthrough",
        )
        transformed_data = transformer.fit_transform(pd.DataFrame(l))
        transformed_data = pd.DataFrame(
            transformed_data, columns=transformer.get_feature_names_out()
        )
        transformed_data.to_csv("transformed.csv")
        print(transformed_data.describe())
