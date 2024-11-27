import requests

url = "http://127.0.0.1:5000/"
data = [
    "SELECT * FROM CONTESTANTS;",
    "SELECT * FROM AREA_CODE_STATE;",
]
response = requests.post(url, json=data)
print(response)
