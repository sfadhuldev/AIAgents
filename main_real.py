import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()


def get_weather(city: str):
    """Get weather for a given city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is missing from the .env file"}

    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return {
                "error": data.get("message", "Weather API request failed")
            }

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"]
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error while getting weather: {e}"}


def get_location():
    """Get user's current location. Use this when the user asks about weather."""

    try:
        response = requests.get(
            "https://ipapi.co/json/",
            headers={"User-Agent": "weather-agent/0.1"},
            timeout=10
        )

        if response.status_code != 200:
            return "Rome, Italy"

        data = response.json()

        city = data.get("city")
        country = data.get("country_name")

        if not city or not country:
            return "Rome, Italy"

        return f"{city}, {country}"

    except requests.exceptions.RequestException:
        return "Rome, Italy"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

system_prompt = """
You are a helpful weather assistant.

YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location:
   - First call get_location()
   - Then call get_weather(city)

2. If the user provides a city:
   - Call get_weather(city) directly.

When you receive weather data from a tool, answer the user in a short clear sentence.
"""

agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt
)


if __name__ == "__main__":
    user_query = input("Enter your query: ")

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            }
        )

        print(response["messages"][-1].content)

    except Exception as e:
        print("The AI model failed.")
        print("Reason:", e)
        print("Most likely you reached Gemini API quota limit.")