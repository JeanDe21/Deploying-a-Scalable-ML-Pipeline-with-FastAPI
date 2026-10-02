
import requests


# Send a GET request.
r = requests.get("http://127.0.0.1:8000")

print("GET Status Code:", r.status_code)
print("GET Result:", r.json())


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


# Send a POST request.
r = requests.post(
    "http://127.0.0.1:8000/data/",
    json=data
)

print("POST Status Code:", r.status_code)
print("POST Result:", r.json())
