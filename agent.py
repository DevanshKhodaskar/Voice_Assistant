from dotenv import load_dotenv
import asyncio
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import groq, noise_cancellation, silero, elevenlabs
import os

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)

# Enhanced instructions for English coaching
COACH_INSTRUCTIONS = """
You are an English speaking coach. Help improve grammar, vocabulary, and fluency through short, natural conversations.

RULES:
- Have a normal conversation with the user and encourage it.
- Keep ALL responses under 30 words
- Correct gently: "Try: [correct version]" 
- Ask one follow-up question if needed only.
- Keep it like normal conversations try to carry on canversation
- talk like a friend
- Be encouraging but brief
- Focus on one correction at a time


KEEP IT SHORT AND HELPFUL.
"""

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        vad=silero.VAD.load(),  # Keep it simple like your working version
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en",
        ),
        llm=groq.LLM(
            model="qwen/qwen3-32b",
        ),
        tts=elevenlabs.TTS(
            voice_id="pqHfZKP75CvOlQylNhV4",
            model="eleven_multilingual_v2"
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=COACH_INSTRUCTIONS
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting
    await session.generate_reply(
        instructions="Say: 'Hi! I'm your English coach. How are you today?"
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))