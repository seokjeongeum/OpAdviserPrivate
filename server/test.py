import requests


def tune():
    print(
        requests.post(
            "http://127.0.0.1:5000/tune",
            json=[
                "SELECT * FROM CONTESTANTS;",
                "SELECT * FROM AREA_CODE_STATE;",
            ],
        )
    )


def query():
    print(
        requests.post(
            "http://127.0.0.1:5000/query",
            json="SELECT * FROM CONTESTANTS;",
        )
    )


query()
tune()
