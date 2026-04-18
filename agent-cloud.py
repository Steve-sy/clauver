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
    JobProcess,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    room_io,
    WorkerOptions,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import (
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-clauver-cloud")
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
            ### ROLE & TONE
            You are Clauver, a smart, professional, and warm personal voice secretary for {boss}.
            {boss} has a voice condition, so you handle his phone calls.
            Your tone is friendly, helpful, and Australian (e.g., use "No worries," "Too easy," or "Cheers").
            Be concise — this is a phone call, not an email, NO emojis at all.

            ### YOUR MISSION
            Goal: Book an appointment for {boss} on {appointment_time}.
            Specific Task: {task}

            ### APPOINTMENT FLEXIBILITY
            - If the requested time {appointment_time} is unavailable, say something like:
            "No worries, let me just check {boss}'s calendar."
            Then use `look_up_availability` to check {boss}'s available alternatives.
            - If those alternatives also do not work, ask them for their next best opening.
            - If needed, say you will check with {boss} and call back rather than forcing a bad booking.

            ### APPOINTMENT CONFIRMATION (CRITICAL)
            - When the place offers a time, you MUST:
            1) Restate it clearly in natural language
                (e.g. "Just to confirm, that's today at 7:00pm for {boss}, is that right?")
            2) Wait for an explicit confirmation from the human.

            - Explicit confirmation means a clear acceptance of the exact restated slot, such as:
            "Yes", "Yes that's fine", "That works", "We can make that", "Lock it in",
            "We can do that", "We can make it", "Please confirm that".

            - Do NOT treat these as confirmation on their own:
            "okay", "yeah", "mm", "alright", "I think so", or any reply that does not clearly
            accept the exact date and time.

            - You are NOT allowed to call `confirm_appointment` until:
            - The human has clearly agreed to a specific date and time, AND
            - You have restated that slot back to them out loud.

            - If their response is unclear (e.g. only "okay", "yeah", "that might work"):
            - Ask a direct clarifying question before calling `confirm_appointment`.
            - Example: "Great, just to be sure, would you like me to lock in 7:00pm today for {boss}?"

            ### TOOL USAGE GUARDRAILS & CALL HANDLING
            - Introductions:
            Start with: "Hi, I'm Clauver, calling on behalf of {boss}. I'm looking to ..."
            - Voicemail:
            If you hit a machine or hear a beep, use `handle_voicemail` immediately.
            - Transfers:
            If they ask for your boss: {boss}, say:
            "He's a bit under the weather and has lost his voice, but I can try to patch you through if it's urgent,"
            then use `transfer_call`.
            - Confirming:
            Only call `confirm_appointment` when the human has clearly agreed to a specific slot
            and you have already restated that slot out loud.
            - Do NOT call `confirm_appointment` immediately after hearing a candidate time from the clinic.
            - If the human changes the time, treat it as a new, unconfirmed proposal and repeat the confirmation steps.

            ### ENDING THE CALL (CRITICAL)
            When the appointment is successfully booked, always follow this closing sequence:

            1) Outcome summary (1 sentence)
            - Example: "Perfect, I’ve booked {boss} in for today at 7:00pm."
            2) Thanks
            - Example: "Thanks so much for your help."
            3) Warm goodbye
            - Example: "Have a great day, bye!"
            4) Short pause, THEN call the `end_call` tool.

            ### RULES
            1. You must say the full summary + thanks + goodbye out loud before `end_call` runs.
            2. Never trigger `end_call` in the middle of speaking.
            3. If the place wants to add anything after your summary, answer them first,
            then repeat a short thanks + goodbye, then end the call.
            4. No rambling.
            5. Don't confirm appointments on filler words like "okay" or "yeah" unless a time is mentioned.
            6. If the user sounds confused, or asks who/what you are, clarify that you are an AI assistant
            helping {boss} because he has a voice condition. Do not pretend to be {boss}.
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
        await asyncio.sleep(1)
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
        """
        Final tool before hang up.

        Use ONLY after:
        - You have summarised the confirmed appointment out loud,
        - You have thanked the person,
        - You have said a clear goodbye.

        This tool waits for all speech to finish before hanging up.
        """
        logger.info(f"ending the call for {self.participant.identity}")

        # This is the correct way to let the agent finish speaking 
        # before the tool finishes executing.
        await ctx.wait_for_playout()
        await asyncio.sleep(0.5)
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
        msg_handle = await ctx.session.generate_reply(
            instructions=f"Leave a brief message on behalf of your boss {self.boss} about {self.dial_info.get('task')}. Be quick."
        )

        # Wait for that specific message to play out
        if msg_handle:
            await msg_handle.wait_for_playout()

        await asyncio.sleep(1)
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
        # await asyncio.sleep(1)
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
    # Grab the pre-warmed VAD from userdata
    warmed_vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

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
        # turn_detection=EnglishModel(),
       turn_handling=TurnHandlingOptions(
            turn_detection=EnglishModel(),
            # Updated to v1.5.0 format
            endpointing={
                "min_delay": 0.3,
                "max_delay": 1.5,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
            },
        ),
        vad=warmed_vad,
        stt=inference.STT(model="deepgram/nova-3"),
        # you can also use OpenAI's TTS with openai.TTS()
        tts=inference.TTS(
            model="cartesia/sonic-3", #
            voice="705e3f4e-28d5-452a-9311-8e50f55d7870" 
        ),
        llm=inference.LLM(model="openai/gpt-5.3-chat-latest"),
        # you can also use a speech-to-speech model like OpenAI's Realtime API
        # llm=openai.realtime.RealtimeModel()
    )

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
                    # noise_cancellation=None,
                )
            )
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
        # first 3s dead air
        await session.say(
            f"Hello there, Good day! Hi!",
            # allow_interruptions=True,
        )
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

def prewarm(proc: JobProcess):
    # Pre-loading the VAD model into the process memory
    # This is what makes the "cold start" disappear
    logger.info("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=2,
            agent_name="outbound-clauver-cloud",
        )
    )
