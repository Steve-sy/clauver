# This is an example of an outbound calling agent, serverless using LiveKit Cloud, no openAi, deepgram or cartesia API keys required.
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
    inference, # Added for cloud-managed AI
)
from livekit.plugins import (
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            You are a scheduling assistant for a dental practice. Your interface with user will be voice.
            You will be on a call with a patient who has an upcoming appointment. Your goal is to confirm the appointment details.
            As a customer service representative, you will be polite and professional at all times. Allow user to end the conversation.

            When the user would like to be transferred to a human agent, first confirm with them. upon confirmation, use the transfer_call tool.
            The customer's name is {name}. His appointment is on {appointment_time}.
            """
        )
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

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
        logger.info(f"ending the call for {self.participant.identity}")
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def look_up_availability(self, ctx: RunContext, date: str):
        logger.info(f"looking up availability for {self.participant.identity} on {date}")
        await asyncio.sleep(3)
        return {"available_times": ["1pm", "2pm", "3pm"]}

    @function_tool()
    async def confirm_appointment(self, ctx: RunContext, date: str, time: str):
        logger.info(f"confirming appointment for {self.participant.identity} on {date} at {time}")
        return "reservation confirmed"

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]

    agent = OutboundCaller(
        name="Mustafa",
        appointment_time="next Tuesday at 3pm",
        dial_info=dial_info,
    )

    # REFACTORED: Using LiveKit Cloud Inference
    session = AgentSession(
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=inference.STT(model="deepgram/nova-3"), #
        tts=inference.TTS(
            model="cartesia/sonic-3", #
            voice="705e3f4e-28d5-452a-9311-8e50f55d7870" 
        ),
        llm=inference.LLM(model="openai/gpt-5.4-mini"), #
    )

    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
            )
        )

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
            agent_name="outbound-caller",
        )
    )