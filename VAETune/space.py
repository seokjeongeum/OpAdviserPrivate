from collections import defaultdict
import json
import pprint
import re
from typing import List


class Hyperparameter:
    def __init__(
        self, name: str, param_type: str, param_range: List[int], default: int
    ):
        self.name = name
        self.param_type = param_type
        self.param_range = param_range
        self.default = default

    def __repr__(self) -> str:
        return (
            f"Hyperparameter(name='{self.name}', type='{self.param_type}', "
            f"range={self.param_range}, default={self.default})"
        )


CONFIG_BLOCK_PATTERN = re.compile(
    r"Configuration space object:\s*"
    r"Hyperparameters:(.*?)(?=Configuration space object:|$)",
    re.DOTALL,  # Enables matching across multiple lines
)
HYPERPARAMETER_PATTERN = re.compile(
    r"\s*(\w+), Type: (\w+), Range: \[(\d+), (\d+)\], Default: (\d+)"
)
if __name__ == "__main__":
    s = "sbrw_smac"
    with open(f"repo/history_{s}.json") as f:
        j = json.load(f)["data"]
        c = sorted(j, key=lambda x: x["external_metrics"].get("tps", 0))[-1]
        pprint.pprint(c["external_metrics"])
    with open(f"repo/history_sysbench_smac.json") as f:
        j = json.load(f)["data"]
        c = sorted(j, key=lambda x: x["external_metrics"].get("tps", 0))[-1]
        pprint.pprint(c["external_metrics"])
    with open(f"logs/DBTune-{s}.log") as f:
        c = c["configuration"]
        config_text = f.read()
        config_blocks = CONFIG_BLOCK_PATTERN.findall(config_text)
        d = defaultdict(list)
        for block in config_blocks:
            hyperparameters = []
            for match in HYPERPARAMETER_PATTERN.findall(block):
                name, param_type, range_start, range_end, default = match
                if param_type != "UniformInteger":
                    pprint.pprint(param_type)
                if int(range_start) <= c[name] <= int(range_end):
                    d[name].append(1)
                else:
                    d[name].append(0)
                hyperparameters.append(
                    Hyperparameter(
                        name=name,
                        param_type=param_type,
                        param_range=[int(range_start), int(range_end)],
                        default=int(default),
                    )
                )
            # pprint.pprint(hyperparameters)
    pprint.pprint(
        sorted(
            map(lambda x: (x[0], (len(x[1]), sum(x[1]) / len(x[1]))), d.items()),
            key=lambda x: x[1],
        )
    )
    print(d)
    # pprint.pprint(
    #     list(
    #         map(lambda x: (x[0], x[1]["external_metrics"].get("tps", 0)), enumerate(j))
    #     )
    # )
