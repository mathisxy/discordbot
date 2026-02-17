from edgygraph import Graph, START, Node, State, Shared
import discord
from discord.ext import commands
import os
from rich import print as rprint
import fastmcp
from typing import Protocol
from llmir import AIMessage, AIRoles, AIChunkText

from edgynodes.discordvoice import JoinVoiceChannelNode, StartRecordVoiceNode, AwaitVoiceStopVADNode, AwaitVoiceStartVADNode, StopRecordVoiceNode, STTMistralNode
from edgynodes.discordvoice_llm import TranscriptionsToAINode, PiperTTSNode
from edgynodes.discord_llm import BuildChatNode, RespondNode
from edgynodes.llm import LLMMistralNode, SaveNewMessagesNode, AddMCPToolsNode, AddToolsNode, GetNextToolCallResultNode, ExtractNewToolCallsNode, IntegrateMCPToolResultsNode, IntegrateToolResultsNode, AddMessageNode
from edgynodes.discordtmp import ClearTmpDiscordMessagesNode, TemporaryMessageController
from edgynodes.discord import StartTypingNode, StopTypingNode
import edgynodes as e

from tools import leave_voice_channel
from mcp_client import get_log_handler, get_progress_handler

### STATES

class MyStateProtocol(e.discordvoice.StateProtocol, e.discord.StateProtocol, e.llm.StateProtocol, e.discordtmp.StateProtocol, Protocol):
    pass

class MySharedProtocol(e.discordvoice.SharedProtocol, e.discord.SharedProtocol, e.llm.SharedProtocol, e.discordtmp.SharedProtocol, Protocol):
    pass

class MyState(State):
    discord: e.discord.StateAttribute
    discordvoice: e.discordvoice.StateAttribute
    llm: e.llm.StateAttribute
    discordtmp: e.discordtmp.StateAttribute

class MyShared(Shared):
    discord: e.discord.SharedAttribute
    discordvoice: e.discordvoice.SharedAttribute
    llm: e.llm.SharedAttribute
    discordtmp: e.discordtmp.SharedAttribute


### NODES


class SendSinkedVoiceNode(Node[e.discordvoice.StateProtocol, e.discordvoice.SharedProtocol]):

    async def run(self, state: e.discordvoice.StateProtocol, shared: e.discordvoice.SharedProtocol) -> None:
        
        async with shared.lock:
            if not shared.discordvoice.sink:
                raise RuntimeError("No voice sink available to send recordings.")

            text_channel = shared.discordvoice.text_channel
            if not text_channel:
                raise RuntimeError("No text channel available to send recordings.")

            sink = shared.discordvoice.sink

        recorded_users = [  # A list of recorded users
            f"<@{user_id}>"
            for user_id, audio in sink.audio_data.items()
        ]

        for user_id, audio in sink.audio_data.items():
            file = discord.File(audio.file, f"recording-{user_id}.{sink.encoding}")

            await text_channel.send(
                content=f"Recording of <@{user_id}>",
                file=file
            )


async def handle_voice(channel: discord.VoiceChannel, text_channel: discord.abc.Messageable, bot: commands.Bot) -> None:

    # if bot.voice_clients:
    #     voice_client = bot.voice_clients[0]

    #     # Schon im richtigen Channel
    #     if voice_client.channel != channel:
            
    #         # Sonst rüberziehen
    #         await voice_client.move_to(channel) # type: ignore

    # else:
    #     # Bot ist noch nirgends verbunden
    #     voice_client = await channel.connect()


    temporary_message_controller = TemporaryMessageController(text_channel)


    log_handler = get_log_handler(temporary_message_controller)
    progress_handler = get_progress_handler(temporary_message_controller)


    mcp_client = fastmcp.Client("http://localhost:8001/mcp", log_handler=log_handler, progress_handler=progress_handler)

    state = MyState(
        discordvoice=e.discordvoice.StateAttribute(),
        discord=e.discord.StateAttribute(),
        llm=e.llm.StateAttribute(),
        discordtmp=e.discordtmp.StateAttribute()
    )
    shared = MyShared(
        discordvoice=e.discordvoice.SharedAttribute(
            channel=channel,
            client=None,
            text_channel=text_channel
        ),
        discord=e.discord.SharedAttribute(
            text_channel=text_channel,
            bot=bot
        ),
        llm=e.llm.SharedAttribute(),
        discordtmp=e.discordtmp.SharedAttribute(
            controller=temporary_message_controller
        )
    )

    join = JoinVoiceChannelNode()
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    add_message = AddMessageNode(AIMessage(
        role=AIRoles.SYSTEM,
        chunks=[
            AIChunkText(
                text="Du bist Emanuel. Deine Nachrichten werden in Sprache ausgegeben im Discord Gruppen Voice Channel. " \
                "Nutze deshalb NIEMALS Emojis oder Smileys und halte dich kurz. Überprüfe deine Antworten und entferne immer alle Emojies!"
            )
        ]
    ))
    start_record = StartRecordVoiceNode(sink_factory=lambda: e.discordvoice.VADWaveSink(), delay=0.5)
    stop_record = StopRecordVoiceNode()
    wait_voice = AwaitVoiceStartVADNode()
    wait_silence = AwaitVoiceStopVADNode(silence_timeout=1.5)
    stt = STTMistralNode(api_key=os.getenv("MISTRAL_API_KEY", "")) # 0.5s
    transcription_to_ai = TranscriptionsToAINode()
    build_chat = BuildChatNode()
    llm = LLMMistralNode(api_key=os.getenv("MISTRAL_API_KEY", ""), model="mistral-medium-latest", stream=True)
    respond = RespondNode()
    respond_tool_call_results = RespondNode()
    save_messages = SaveNewMessagesNode()
    save_for_new_turn = SaveNewMessagesNode()
    save_for_reset = SaveNewMessagesNode()
    save_transcription = SaveNewMessagesNode()
    mcp_tools = AddMCPToolsNode(mcp_client)
    tools = AddToolsNode([leave_voice_channel])
    get_new_tool_calls = ExtractNewToolCallsNode()
    get_next_tool_call_result = GetNextToolCallResultNode()
    integrate_tool_call_results = IntegrateToolResultsNode()
    integrate_mcp_tool_call_results = IntegrateMCPToolResultsNode()
    clear_tmp_discord_messages = ClearTmpDiscordMessagesNode()

    piper = PiperTTSNode() # ("piper/de_DE-eva_k-x_low.onnx", "piper/de_DE-eva_k-x_low.onnx.json"))


    await Graph[MyStateProtocol, MySharedProtocol](
        edges=[
            (
                START,
                mcp_tools,
            ),
            (
                mcp_tools,
                tools,
            ),
            (
                tools,
                add_message
            ),
            (
                add_message,
                build_chat
            ),
            (
                build_chat,
                join
            ),
            (
                join,
                start_record
            ),
            (
                start_record,
                wait_voice,
            ),
            (
                wait_voice,
                wait_silence,
            ),
            (
                wait_silence,
                stop_record
            ),
            (
                stop_record,
                stt
            ),
            (
                stt,
                transcription_to_ai
            ),
            (
                transcription_to_ai,
                lambda st, sh: save_transcription if st.llm.new_messages else start_record
            ),
            (
                save_transcription,
                start_typing
            ),
            (
                start_typing,
                llm
            ),
            (
                llm,
            #     respond
            # ),
            # (
            #     respond,
                piper
            ),
            (
                piper,
                get_new_tool_calls,
            ),
            (
                get_new_tool_calls,
                save_messages
            ),
            (
                save_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results
            ),
            (
                get_next_tool_call_result,
                clear_tmp_discord_messages
            ),
            (
                clear_tmp_discord_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results
            ),
            (
                integrate_mcp_tool_call_results,
                integrate_tool_call_results
            ),
            (
                integrate_tool_call_results,
                respond_tool_call_results
            ),
            (
                respond_tool_call_results,
                lambda st, sh: save_for_new_turn if st.llm.new_messages else save_for_reset
            ),
            (
                save_for_new_turn,
                llm
            ),
            (
                save_for_reset,
                stop_typing
            ),
            (
                stop_typing,
                start_record
            )
        ]
    )(state, shared)