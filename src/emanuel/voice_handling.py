from edgygraph import Graph, START, Node, StateProtocol, SharedProtocol, END, State, Shared
from edgygraph.graph_hooks import NodePrintHook
import discord
from discord.ext import commands
import os
from typing import Protocol, Literal
from llmir import AIMessage, AIRoles, AIChunkText
from piper import SynthesisConfig

from edgynodes.discordvoice.nodes.channel import JoinVoiceChannelNode
from edgynodes.discordvoice.nodes.recording import StartRecordVoiceNode, StopRecordVoiceNode
from edgynodes.discordvoice.nodes.vad import VADWaveSink, AwaitVoiceStartVADNode, AwaitVoiceStopVADNode
from edgynodes.discordvoice.nodes.stt import STTMistralNode
from edgynodes.discordvoice.nodes.interrupt import SetInterruptNode, ClearInterruptNode
from edgynodes.discordvoice_llm.nodes.transcriptions_to_ai import TranscriptionsToAINode
from edgynodes.discordvoice_llm.nodes.tts.piper import PiperTTSNode
from edgynodes.discord_llm.nodes.build_chat import BuildChatNode
from edgynodes.discord_llm.nodes.respond import RespondNode
from edgynodes.llm.nodes.messages import SaveNewMessagesNode, AddMessageNode
from edgynodes.llm.nodes.tools import AddToolsNode, GetNextToolCallResultNode, ExtractNewToolCallsNode, IntegrateMCPToolResultsNode, IntegrateToolResultsNode, ToolContext
from edgynodes.llm.nodes.formatting import StripFormattingsNode
from edgynodes.discordtmp.nodes.clear import ClearTmpDiscordMessagesNode
from edgynodes.discordtmp import TemporaryMessageController
from edgynodes.discord.nodes.typing import StartTypingNode, StopTypingNode

import edgynodes as e

from tools import leave_voice_channel
from get_nodes import get_llm_node, get_mcp_nodes, get_voice_system_message_node

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


### TOOLS

syn_config = SynthesisConfig()

def set_voice(speaker: Literal["amused", "angry", "disgusted", "drunk", "neutral", "sleepy", "surprised", "whisper"]) -> str:
    ids = {
        "amused": 0,
        "angry": 1,
        "disgusted": 2,
        "drunk": 3,
        "neutral": 4,
        "sleepy": 5,
        "surprised": 6,
        "whisper": 7,
    }
    syn_config.speaker_id = ids[speaker]

    return "Voice set to " + speaker 



### NODES

class DebugNode(Node[StateProtocol, SharedProtocol]):

    def __init__(self, name: str = "DebugNode") -> None:
        self.name = name

    async def __call__(self, state: StateProtocol, shared: SharedProtocol) -> None:
        print(f"DEBUG NODE: {self.name}")


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

    # return


    temporary_message_controller = TemporaryMessageController(text_channel)


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

    async def on_end(state: MyStateProtocol, shared: MySharedProtocol) -> None:
        await leave_voice_channel(ToolContext[MyStateProtocol, MySharedProtocol](state=state, shared=shared))

    join = JoinVoiceChannelNode()
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    stop_typing_after_error = StopTypingNode()
    add_message = get_voice_system_message_node()
    start_record = StartRecordVoiceNode(sink_factory=lambda: VADWaveSink(), delay=0.5)
    stop_record = StopRecordVoiceNode()
    wait_voice = AwaitVoiceStartVADNode()
    start_record_for_interrupt = StartRecordVoiceNode(sink_factory=lambda: VADWaveSink(3), delay=0.5)
    wait_voice_for_interrupt = AwaitVoiceStartVADNode()
    interrupt = SetInterruptNode()
    clear_interrupt = ClearInterruptNode()

    wait_silence = AwaitVoiceStopVADNode(silence_timeout=1.5)
    stt = STTMistralNode(api_key=os.getenv("MISTRAL_API_KEY", "")) # 0.5s
    transcription_to_ai = TranscriptionsToAINode()
    build_chat = BuildChatNode(limit=1)
    llm = get_llm_node()
    respond = RespondNode(filter=lambda m, c: not isinstance(c, AIChunkText))
    respond_tool_call_results = RespondNode()
    save_messages = SaveNewMessagesNode()
    save_for_new_turn = SaveNewMessagesNode()
    save_for_reset = SaveNewMessagesNode()
    save_transcription = SaveNewMessagesNode()
    add_mcp_tools = get_mcp_nodes(temporary_message_controller)
    tools = AddToolsNode([leave_voice_channel])
    get_new_tool_calls = ExtractNewToolCallsNode()
    get_next_tool_call_result = GetNextToolCallResultNode()
    integrate_tool_call_results = IntegrateToolResultsNode()
    integrate_mcp_tool_call_results = IntegrateMCPToolResultsNode()
    clear_tmp_discord_messages = ClearTmpDiscordMessagesNode()
    debug1 = DebugNode("_________----------------------------------------_____________---__-__-___---__-____-__-----_-___---_")
    debug2 = DebugNode("2")

    initial = (*add_mcp_tools, tools, add_message, build_chat, join)

    tts = PiperTTSNode(syn_config=syn_config) # ("piper/de_DE-eva_k-x_low.onnx", "piper/de_DE-eva_k-x_low.onnx.json"))

    strip_formattings = StripFormattingsNode()


    await Graph[MyStateProtocol, MySharedProtocol](
        hooks=[
            #InteractiveDebugHook(),
            NodePrintHook()
        ],
        edges=[
            (
                START,
                *initial,
                start_record,
                wait_voice,
                wait_silence,
                stop_record,
                clear_interrupt,
                stt,
                transcription_to_ai,
                lambda st, sh: [save_transcription, start_typing] if st.llm.new_messages else start_record,

                start_typing,
                llm,
                strip_formattings,
                tts,
                respond,
                get_new_tool_calls,
                save_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results,

                get_next_tool_call_result,
                clear_tmp_discord_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results,

                integrate_mcp_tool_call_results,
                integrate_tool_call_results,
                respond_tool_call_results,
                lambda st, sh: save_for_new_turn if st.llm.new_messages else save_for_reset,

                save_for_new_turn,
                -llm,

                save_for_reset,
                stop_typing,
                wait_voice,

                (tts, Exception),
                wait_silence,

                Exception,
                stop_typing_after_error,

                stop_typing_after_error,
                lambda st, sh: on_end(st, sh),

                END
            ),
            (
                start_typing,
                debug1,
                start_record_for_interrupt,
                wait_voice_for_interrupt,
                interrupt,
                
                clear_interrupt
            )]
    )(state, shared)