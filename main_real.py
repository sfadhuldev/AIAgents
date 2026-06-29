import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()


def get_user_location_data():
    try:
        response = requests.get(
            "https://ipapi.co/json/",
            headers={"User-Agent": "weather-agent/0.1"},
            timeout=10
        )

        if response.status_code != 200:
            return None

        return response.json()

    except requests.exceptions.RequestException:
        return None


def get_location():
    """Get user's current location. Use this when the user asks about weather."""
    data = get_user_location_data()

    if not data:
        return "Manama,BH"

    city = data.get("city")
    country_code = data.get("country_code")

    if not city or not country_code:
        return "Manama,BH"

    return f"{city},{country_code}"


def get_weather(city: str = ""):
    """Get weather for a given city. If city is empty, use user's current location coordinates."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is missing from the .env file"}

    base_url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        # If Ollama sends empty city, use coordinates
        if not city or city.strip() == "":
            location_data = get_user_location_data()

            if location_data:
                lat = location_data.get("latitude")
                lon = location_data.get("longitude")

                if lat and lon:
                    params = {
                        "lat": lat,
                        "lon": lon,
                        "appid": api_key,
                        "units": "metric"
                    }
                else:
                    params = {
                        "q": "Manama,BH",
                        "appid": api_key,
                        "units": "metric"
                    }
            else:
                params = {
                    "q": "Manama,BH",
                    "appid": api_key,
                    "units": "metric"
                }

        else:
            city = city.strip()

            # Fix Bahrain small-area issue
            if "Tūblī" in city or "Tubli" in city:
                city = "Manama,BH"

            params = {
                "q": city,
                "appid": api_key,
                "units": "metric"
            }

        print("Weather params:", params)

        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return {
                "error": data.get("message", "Weather API request failed"),
                "params_used": params
            }

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature_celsius": data["main"]["temp"],
            "condition": data["weather"][0]["description"]
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error while getting weather: {e}"}


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

ollama = ChatOllama(
    model="llama3.2",
    temperature=0.7,
)


system_prompt = """
You are a helpful weather assistant.

YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location:
   - Call get_weather with an empty city string.

2. If the user provides a city:
   - Call get_weather(city) directly.

3. The weather tool returns temperature in Celsius.

4. Answer in a short, clear sentence.
"""


def create_weather_agent(model):
    return create_agent(
        model=model,
        tools=[get_weather, get_location],
        system_prompt=system_prompt
    )


if __name__ == "__main__":
    user_query = input("Enter your query: ")

    try:
        print("Using Ollama...")

        agent = create_weather_agent(ollama)

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

    except Exception as e:
        print("Ollama failed.")
        print("Reason:", e)
        print("Switching to Gemini...")

        try:
            agent = create_weather_agent(gemini)

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

        except Exception as e:
            print("Gemini also failed.")
            print("Reason:", e)
            raise

    print(response["messages"][-1].content)