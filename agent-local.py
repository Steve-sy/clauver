# This is an example of an outbound calling agent. Locally you will need openAi, deepgram and cartesia API keys.
from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("clauver-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        task: str,
        boss: str = "Max", # the owner name
        appointment_time: str,
        flexibility_time: Any,
        dial_info: dict[str, Any],
    ):
        # The core persona logic - using the task passed from Metadata
        super().__init__(
            instructions=f"""
            You are Clauver, a smart and professional personal voice secretary for {boss}. 
            {boss} has a health condition that makes his voice weak, so you are speaking on his behalf on a phone call.
            Your goal is to make appointment for your {boss} on {appointment_time}.
            Be polite, clear, and use a friendly Australian tone. Allow user to end the conversation.
            
            If the user says a time is unavailable, use look_up_availability. If those times also don't work, ask the user for their next best opening and tell them you will check with {boss} and call back

            When the user would like to be transferred to a human agent, first confirm with them. upon confirmation, use the transfer_call tool.

            YOUR SPECIFIC MISSION FOR THIS CALL:
            {task}

            Rules:
            1. Introduce yourself immediately: "Hi, I'm Clauver, calling on behalf of {boss}."
            2. Be concise. Don't ramble.
            3. If you encounter a voicemail greeting or hear a beep, use the handle_voicemail tool to leave a short message and exit the call.
            4. If you hear an automated menu, you can navigate it by speaking the numbers or options. For example, say 'One' to select the first option.
            5. If the user asks to speak to {boss}, explain he's sick but you will transfer the call.
            6. Use the end_call tool when the objective is met or the person hangs up.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None

        self.dial_info = dial_info

        self.flexibility_time = flexibility_time
        self.boss = boss
        self.appointment_time = appointment_time

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human agent, called after confirming with the user"""

        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        await ctx.session.generate_reply(
            instructions="let the user know you'll be transferring them"
        )

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call."
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")

        # This is the correct way to let the agent finish speaking 
        # before the tool finishes executing.
        await ctx.wait_for_playout()

        await self.hangup()

    @function_tool()
    async def handle_voicemail(self, ctx: RunContext):
        """
        Called when the call reaches a voicemail greeting or an answering machine beep.
        Use this tool to either leave a concise message or hang up if no message is needed.
        """
        logger.info(f"Voicemail detected for {self.participant.identity}")
        
        # Check if we should leave a message based on the task
        # We tell the agent to generate a final goodbye/message
        await ctx.session.generate_reply(
            instructions=f"You have reached voicemail. Leave a 1-sentence message for {self.boss} regarding the task: '{self.dial_info.get('task')}', then say goodbye."
        )

        # Give the TTS enough time to finish speaking the message
        # We use a dynamic wait or a safe buffer
        await asyncio.sleep(6) 
        
        logger.info("Message left, ending call.")
        await self.hangup()

    @function_tool()
    async def look_up_availability(
        self,
        ctx: RunContext,
        date: str,
    ):
        """Called when the current appointment_time is unavailable. 
        Checks the boss's backup options and flexibility.

        Args:
            date: The date of the appointment to check availability for
        """
        logger.info(
            f"looking up availability for {self.participant.identity} on {date}"
        )
        await asyncio.sleep(3)
        if isinstance(self.flexibility_time, list):
            return {"available_slots": self.flexibility_time}
        
        return {
        "general_availability_rules": self.flexibility_time,
        "instructions": "If these times don't work, ask the recipient for their next opening."
    }

    @function_tool()
    async def confirm_appointment(
        self,
        ctx: RunContext,
        date: str,
        time: str,
    ):
        """Called when the user confirms the appointment on a specific date.
        Use this tool only when they are certain about the date and time.

        Args:
            date: The date of the appointment
            time: The time of the appointment
        """
        logger.info(
            f"confirming appointment for {self.participant.identity} on {date} at {time}"
        )
        return "reservation confirmed"

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to Clauver Room: {ctx.room.name}")
    await ctx.connect()

    # --- METADATA PARSING LOGIC ---
    # This pulls the 'task' and 'phone_number' from your CLI command
    # when dispatching the agent, we'll pass it the approriate info to dial the user
    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - transfer_to: the phone number to transfer the call to when requested
    dial_info = json.loads(ctx.job.metadata)
    phone_number = dial_info.get("phone_number")
    # Default task if you forget to send one
    task = dial_info.get("task", "Introduce yourself and ask how you can help Max.")
    participant_identity = phone_number = dial_info["phone_number"]
    appointment_time = dial_info.get("appointment_time", "today at 3pm")
    flexibility_time = dial_info.get("flexibility_time", ["anytime this week"])
    # look up the user's phone number and call details
    agent = OutboundCaller(
        task=task,
        appointment_time=appointment_time,
        flexibility_time=flexibility_time,
        dial_info=dial_info,
    )

    # the following uses gpt-4o, Deepgram and Cartesia
    session = AgentSession(
        # SIP TUNING: Wait longer for the human to finish speaking
        # turn_detection=EnglishModel(min_endpointing_delay=0.8),
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        # you can also use OpenAI's TTS with openai.TTS()
        # Aussie Voice ID integrated here
        tts=cartesia.TTS(
            model="sonic-3",
            voice="a4a16c5e-5902-4732-b9b6-2a48efd2e11b" 
        ),
        llm=openai.LLM(model="gpt-5.4-mini"), # Stable choice for high logic
        # you can also use a speech-to-speech model like OpenAI's Realtime API
        # llm=openai.realtime.RealtimeModel()
    )

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="clauver",
        )
    )
