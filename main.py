#import libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

#this is fake tool to return the weather
def get_weather(city: str):
    """Get weather for a given city"""
    return {'condition': 'sunny', 'temperature': 25}

#this is fake tool to return the location
def get_location():
    """Get user's current location. Use this when the user asks about weather."""
    return "Rome, Italy"

def get_quality(city: str):
    return {'condition': 'Pollutant'}

# Start Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
#here if you ask the agent what is the weather it will return the get_location
system_prompt = """
You are a helpful weather assistant. 
YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location, you MUST:
   - First call get_location() to find their location
   - Then call get_weather(city) with that location

2. If the user provides a city, call get_weather(city) directly.

"""
# the mind
agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt
)
#take the question
user_query = input("Enter your query: ")

# response1 = llm.invoke("How is the weather in Rome?")
response1 = agent.invoke(
    {"messages": [{'role': 'user',
                   'content': user_query}]})
print(response1['messages'][-1].content)
