from edgygraph import Graph, START, Node
import discord
from discord.ext import commands
import os
from rich import print as rprint
import fastmcp

from edgynodes.discordvoice import JoinVoiceChannelNode, StartRecordVoiceNode, AwaitVoiceStopVADNode, AwaitVoiceStartVADNode, StopRecordVoiceNode, STTMistralNode
from edgynodes.discordvoice_llm import TranscriptionsToAINode
from edgynodes.discord_llm import BuildChatNode, RespondNode
from edgynodes.llm import LLMMistralNode, SaveNewMessagesNode, AddMCPToolsNode, AddToolsNode, GetNextToolCallResultNode, ExtractNewToolCallsNode, IntegrateMCPToolResultsNode, IntegrateToolResultsNode
import edgynodes as e

from tools import leave_voice_channel
from mcp_client import get_log_handler, get_progress_handler

### STATES

class DiscordTextVoiceLLMState(e.discordvoice.State, e.discord.State, e.llm.State, e.discord_llm.State, e.discordvoice_llm.State):
    pass

class DiscordTextVoiceLLMShared(e.discordvoice.Shared, e.discord.Shared, e.llm.Shared, e.discord_llm.Shared, e.discordvoice_llm.Shared):
    pass


### NODES


class SendSinkedVoiceNode(Node[e.discordvoice.State, e.discordvoice.Shared]):

    async def run(self, state: e.discordvoice.State, shared: e.discordvoice.Shared) -> None:
        
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

    if bot.voice_clients:
        voice_client = bot.voice_clients[0]

        # Schon im richtigen Channel
        if voice_client.channel != channel:
            
            # Sonst rüberziehen
            await voice_client.move_to(channel) # type: ignore

    else:
        # Bot ist noch nirgends verbunden
        voice_client = await channel.connect()


    mcp_client = fastmcp.Client("http://localhost:8001/mcp", log_handler=get_log_handler, progress_handler=get_progress_handler)

    state = DiscordTextVoiceLLMState(
        discordvoice=e.discordvoice.StateAttribute(),
        discord=e.discord.StateAttribute(),
        llm=e.llm.StateAttribute()
    )
    shared = DiscordTextVoiceLLMShared(
        discordvoice=e.discordvoice.SharedAttribute(
            channel=channel,
            client=voice_client,
            text_channel=text_channel
        ),
        discord=e.discord.SharedAttribute(
            text_channel=text_channel,
            bot=bot
        ),
        llm=e.llm.SharedAttribute()
    )

    join = JoinVoiceChannelNode()
    start_record = StartRecordVoiceNode(sink_factory=lambda: e.discordvoice.VADWaveSink(), delay=0.5)
    stop_record = StopRecordVoiceNode()
    wait_voice = AwaitVoiceStartVADNode()
    wait_silence = AwaitVoiceStopVADNode(silence_timeout=1.5)
    stt = STTMistralNode(api_key=os.getenv("MISTRAL_API_KEY", ""))
    transcription_to_ai = TranscriptionsToAINode()
    build_chat = BuildChatNode()
    llm = LLMMistralNode(api_key=os.getenv("MISTRAL_API_KEY", ""), model="mistral-medium-latest", stream=True)
    respond = RespondNode()
    save_transcription = SaveNewMessagesNode()
    save_llm_response = SaveNewMessagesNode()
    mcp_tools = AddMCPToolsNode(mcp_client)
    tools = AddToolsNode([leave_voice_channel])
    extract_new_tool_calls = ExtractNewToolCallsNode()
    get_tool_result = GetNextToolCallResultNode()
    integrate_tool_results = IntegrateToolResultsNode()
    integrate_mcp_results = IntegrateMCPToolResultsNode()


    await Graph[DiscordTextVoiceLLMState, DiscordTextVoiceLLMShared](
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
                llm
            ),
            (
                llm,
                extract_new_tool_calls,
            ),
            (
                extract_new_tool_calls,
                lambda st, sh: get_tool_result if sh.llm.new_tool_calls else respond
            ),
            (
                get_tool_result,
                integrate_mcp_results
            ),
            (
                integrate_mcp_results,
                integrate_tool_results
            ),
            (
                integrate_tool_results,
                llm
            ),
            (
                respond,
                save_llm_response,
            ),
            (
                save_llm_response,
                start_record
            ),
        ]
    )(state, shared)