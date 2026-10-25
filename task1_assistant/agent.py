from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from task1_assistant.tools import calculator
from task1_assistant.tools import text_analyzer
root_agent = Agent(
    model="gemini-3.8-flash",
    name='root_agent',
    description='A helpful personal assistant that can search, calculate, and analyze text.',
    instruction="""
    You are a helpful personal assistant.

    Use Google Search when the user asks for current or external information.
    Answer calculations accurately.
    Analyze text when the user asks for text statistics.
    Choose the appropriate tool based on the user's request.
    """,
    tools=[google_search,calculator, text_analyzer],
)
