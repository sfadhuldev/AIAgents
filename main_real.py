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
    """Get weather for a given city.
    Return Fahrenheit for US, Liberia, Myanmar/Burma.
    Return Celsius for all other countries.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is missing from the .env file"}

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    fahrenheit_countries = {"US", "LR", "MM"}

    try:
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

            if "Tūblī" in city or "Tubli" in city:
                city = "Manama,BH"

            params = {
                "q": city,
                "appid": api_key,
                "units": "metric"
            }

        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return {
                "error": data.get("message", "Weather API request failed"),
                "params_used": params
            }

        country = data["sys"]["country"]

        if country in fahrenheit_countries:
            params["units"] = "imperial"
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            unit = "°F"
        else:
            unit = "°C"

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"], 1),
            "unit": unit,
            "condition": data["weather"][0]["description"],
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
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
You are a professional weather assistant.

YOUR WORKFLOW:

1. If the user asks about the weather without specifying a location:
   - Call get_weather("").

2. If the user specifies a city or location:
   - Call get_weather(city).

3. The get_weather tool already returns:
   - city
   - country
   - temperature
   - unit
   - condition
   - feels_like
   - humidity
   - wind_speed
4. Never calculate or convert temperature units yourself.
   Always trust the values returned by the tool.

5. If the tool returns an error, explain it politely.

6. Answer naturally in this style:

The current weather in {city}, {country} is {condition} with a temperature of {temperature}{unit}. It feels like {feels_like}{unit}, the humidity is {humidity}%, and the wind speed is {wind_speed} m/s.

7. Keep the response concise, professional, and natural.

8. Never mention tools, APIs, parameters, JSON, or internal workflow.
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