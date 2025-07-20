import requests

url = "https://yahoo-finance160.p.rapidapi.com/history"

payload = {
	"stock": "ZOMATO.NS",
	"period": "1mo"
}
headers = {
	"x-rapidapi-key": "a85439da56msh17f1223a17bbfc6p10807ejsnf3da307dc41b",
	"x-rapidapi-host": "yahoo-finance160.p.rapidapi.com",
	"Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())