import requests

url = "http://127.0.0.1:5000/predict"

while True:
    text = input("Enter a disaster report (or 'quit' to exit): ")
    if text.lower() == "quit":
        print("Thanks for your report.")
        break

    response = requests.post(url, json={"text": text})
    print(response.json())
