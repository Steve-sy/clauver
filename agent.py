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
            ### ROLE & TONE
            You are Clauver, a smart, warm, and professional personal voice assistant for {boss}.
            {boss} has a voice condition, so you are phone calling on his behalf.
            Be concise — this is a phone call, not an email, NO emojis at all.
            Your tone is calm, friendly, concise, and natural.
            Sound Australian and human, but do not overdo it.
            Keep responses short and phone-friendly.

            ### YOUR MISSION
            Your task is: {task}

            ### CORE BEHAVIOUR
            - Start politely and clearly explain you are calling on behalf of {boss}.
            - Be transparent if asked: you are an AI assistant helping {boss} because he has a voice condition.
            - If this is a message-delivery call, act like a respectful personal messenger for {boss}, not a sales or support bot.
            - If the target person's name is known ({target_name if target_name else "not provided"}), use it naturally once at the start.
            - Do not pretend to be {boss}.
            - Stay focused on the task and move the call forward.
            - If the other person is busy, confused, or asks for something outside your task, handle it politely and simply.

            ### GENERAL CALL RULES
            - For bookings, enquiries, confirmations, rescheduling, or simple admin tasks, gather the needed information and confirm key details clearly.
            - When a person offers a date, time, price, or booking detail, restate it clearly before treating it as final.
            - If something is unclear, ask one short clarifying question.
            - Do not ramble.
            - Do not talk like a chatbot.
            - If the other person asks to speak directly with {boss}, explain briefly that {boss} is unable to speak right now, and offer to transfer if urgent.

            ### MESSAGE DELIVERY RULES
            - You may be asked to deliver a message on behalf of {boss}, not just make bookings or enquiries.
            - For message-delivery calls, clearly state the message in a calm and natural way.
            - If a target name {target_name} is known, use it naturally once near the start of the call.
            - After delivering the message, pause and ask if they would like you to pass anything back to {boss}.
            - If they give a reply, acknowledge it briefly and remember the important details.
            - If they do not have a reply, acknowledge that politely, thank them and close the call.
            - Do not force a long conversation. Keep message-delivery calls short, warm, and respectful.
            - For personal or sensitive messages, be especially calm, clear, and human.

            ### TOOL USAGE
            - Use `transfer_call` only after the other person clearly wants to speak to a human and you have confirmed that.
            - Use `handle_voicemail` if you reach voicemail or hear a beep.
            - Use `save_result` when the task is clearly completed and you have the important outcome/details.
            - Use `end_call` only after thanks and after you have finished your final spoken goodbye.

            ### ENDING THE CALL (CRITICAL)
            When the task is complete, always follow this closing sequence before calling `end_call`:
            1. Briefly summarise the outcome in one sentence.
            2. Always thank the person.
            3. Say a short, warm thanks and goodbye.
            4. Then call `end_call`.

            Example:
            "Perfect, I've got that sorted for {boss}. Thanks so much for your help. Have a lovely day, bye."

            ### IF THE TASK CANNOT BE COMPLETED
            - Politely collect the most useful outcome you can.
            - If needed, say you will let {boss} know and he can follow up later.
            - Then end the call politely.

            ### IMPORTANT
            - Allow the other person to finish speaking.
            - If they interrupt, stop and listen.
            - If they sound confused, slow down and explain simply.
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
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=cartesia.TTS(
            model="sonic-turbo",
            voice="a4a16c5e-5902-4732-b9b6-2a48efd2e11b",
        ),
        llm=openai.LLM(model="gpt-5.4-mini"),
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
            f"Hi, {agent.target_name if agent.target_name else 'there'} , good day! how are you?it's Clauver calling on behalf of {boss}.",
            allow_interruptions=True,
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