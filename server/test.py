import pprint
import requests


def tune():
    pprint.pprint(
        requests.post(
            "http://127.0.0.1:5000/tune",
            json=[
                "SELECT * FROM CONTESTANTS;",
                "SELECT * FROM AREA_CODE_STATE;",
            ],
        ).json()
    )


def query():
    pprint.pprint(
        requests.post(
            "http://127.0.0.1:5000/query",
            json="SELECT * FROM CONTESTANTS;",
        ).json()
    )


def knobs():
    pprint.pprint(
        requests.post(
            "http://127.0.0.1:5000/knobs",
            json="SELECT * FROM CONTESTANTS;",
        ).json()
    )


query()
knobs()
tune()
