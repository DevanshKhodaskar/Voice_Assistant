from dotenv import load_dotenv
import asyncio
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import groq, noise_cancellation, silero
import os

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        vad=silero.VAD.load(),  
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en",
        ),
        llm=groq.LLM(
            model="llama3-8b-8192",
           
        ),
        tts=groq.TTS(
            model="playai-tts",

        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="You are a helpful assistant. Say hi before you start and keep it short."
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Say hi and keep it short."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))