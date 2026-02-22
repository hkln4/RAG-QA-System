from google import genai

client = genai.Client(api_key="AIzaSyDDcD1v62ki_rgagFTqgBt5N4WyxLNqQlk")
models = client.models.list()

for m in models:
    print(m.display_name)