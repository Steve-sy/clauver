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
    TurnHandlingOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("clauver-general-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        boss: str,
        task: str,
        dial_info: dict[str, Any],
        target_name: str | None = None,
    ):
        super().__init__(
            instructions=f"""
            You are Clauver, a warm, clear, professional voice assistant for {boss}.
            {boss} has a voice condition, so you make phone calls and deliver messages on his behalf.
            This is a real phone call: be concise, natural, and calm. No emojis.

            Your job:
            - Deliver {boss}'s message.
            - Give the other person a chance to reply.
            - Pass their reply back to {boss}.
            - End the call politely.

            Call behaviour:
            - Start politely and clearly.
            - If the other person's name is known ({target_name if target_name else "not provided"}), use it once near the start.
            - Clearly say you are calling on behalf of {boss}.
            - Keep your first full reply after the greeting very short (one simple sentence), then add more detail if needed.
            - If asked, say you are an AI assistant helping {boss} because he has a voice condition.
            - Do not pretend to be {boss}.

            Message delivery:
            - State {boss}'s message clearly and naturally.
            - Then ask once if they would like you to pass anything back to {boss}.
            - WAIT for their answer before treating the task as complete.
            - If they give a reply, say thanks I'll pass that along to {boss}.
            - If they have nothing to add, acknowledge that and move to closing.
            - Keep the conversation short, warm, and respectful.

            Tools:
            - Use `save_result` only after:
            - you have delivered the message, and
            - the other person has replied (or clearly has nothing to add).
            - Use `handle_voicemail` if you reach voicemail or a beep.
            - Use `transfer_call` only if they clearly want to speak to a human urgently.
            - Use `end_call` only after your final thanks and goodbye are fully spoken.

            Ending the call:
            1. Briefly summarise what you will pass back to {boss}.
            2. Thank them.
            3. Say a short, warm goodbye.
            4. If they add a final "thank you" or "bye", respond once very briefly.
            5. Then call `end_call`.

            General rules:
            - Let them finish speaking.
            - If they interrupt, stop and listen.
            - Never invent facts that were not said on the call.
            """
        )

        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self.boss = boss
        self.task = task
        self.target_name = target_name
        self.call_result: dict[str, Any] = {
            "status": "unknown",
            "outcome": None,
            "details": [],
        }

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""
        job_ctx = get_job_context()
        await asyncio.sleep(1)
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human after the other person clearly asks for it and confirms they want to be transferred."""
        transfer_to = self.dial_info.get("transfer_to")
        if not transfer_to:
            return "No transfer number is available."

        logger.info(f"transferring call to {transfer_to}")

        await ctx.session.generate_reply(
            instructions="Briefly let the person know you are transferring them now."
        )

        job_ctx = get_job_context()
        try:
            await ctx.wait_for_playout()
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )
            logger.info(f"transferred call to {transfer_to}")
            return "Call transferred successfully."
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="Apologise briefly and say the transfer did not work."
            )
            await ctx.wait_for_playout()
            await self.hangup()
            return "Transfer failed."

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Final tool before hanging up. Use only after your final summary, thanks, and goodbye have been spoken."""
        logger.info(f"ending the call for {self.participant.identity}")

        await ctx.wait_for_playout()
        await asyncio.sleep(0.8)
        await self.hangup()
        return "Call ended."

    @function_tool()
    async def handle_voicemail(self, ctx: RunContext):
        """Called when the call reaches voicemail or an answering machine beep."""
        logger.info(f"detected voicemail for {self.participant.identity}")

        msg_handle = await ctx.session.generate_reply(
            instructions=f"""
            Leave a short voicemail on behalf of {self.boss}.
            State your name, say you are calling on behalf of {self.boss}, mention the purpose briefly,
            ask them to call back if appropriate, then say thank you and goodbye.
            Keep it short.
            """
        )

        if msg_handle:
            await msg_handle.wait_for_playout()

        await asyncio.sleep(0.8)
        await self.hangup()
        return "Voicemail handled."

    @function_tool()
    async def save_result(
        self,
        ctx: RunContext,
        status: str,
        outcome: str,
        details: str = "",
    ):
        """Save the result of the call once the task is complete or clearly blocked.

        Args:
            status: success, failed, voicemail, transferred, or follow_up_needed
            outcome: short summary of what happened
            details: any key details such as time, date, address, booking info, callback request, or next step
        """
        logger.info(
            f"saving result for {self.participant.identity}: status={status}, outcome={outcome}, details={details}"
        )

        self.call_result = {
            "status": status,
            "outcome": outcome,
            "details": details,
        }
        return "Result saved."

    @function_tool()
    async def get_task_context(self, ctx: RunContext):
        """Get the current task and any metadata passed into the call."""
        return {
            "boss": self.boss,
            "task": self.task,
            "dial_info": self.dial_info,
        }


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]
    target_name = dial_info.get("target_name", None)
    boss = dial_info.get("boss", "Max")
    task = dial_info.get(
        "task",
        f"Call on behalf of {boss}, introduce yourself clearly, and help with their request.",
    )

    agent = OutboundCaller(
        target_name=target_name,
        boss=boss,
        task=task,
        dial_info=dial_info,
    )

    session = AgentSession(
        # turn_detection=EnglishModel(),
        turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            # Updated to v1.5.0 format
            endpointing={
                "min_delay": 0.2,
                "max_delay": 1.25,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
            },
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=cartesia.TTS(
            model="sonic-turbo",
            voice="a4a16c5e-5902-4732-b9b6-2a48efd2e11b",
        ),
        llm=openai.LLM(model="gpt-5.3-chat-latest"),
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

        await session.say(
            f"Hi, {agent.target_name if agent.target_name else 'there'}, good day! how are you? I am calling on behalf of {boss}.",
            # allow_interruptions=True,
        )

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
            agent_name="clauver-general",
        )
    )