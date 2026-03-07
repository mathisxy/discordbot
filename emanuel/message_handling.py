from edgygraph import Graph, START, END, Node, State, Shared
from edgygraph.graph_hooks import NodePrintHook
from edgynodes.llm.nodes.core.messages import AddMessageNode
from llmir import AIChunkText, AIMessage, AIRoles
from voice_handling import handle_voice
from logger import setup_logger
from edgynodes.llm import LLMAzureNode, LLMOllamaNode, LLMClaudeNode, ExtractNewToolCallsNode, GetNextToolCallResultNode, IntegrateToolResultsNode, IntegrateMCPToolResultsNode, AddToolsNode, SaveNewMessagesNode, LLMGeminiNode, LLMMistralNode, AddMCPToolsNode, LLMOpenAINode, ToolContext, TurnCounterNode
from edgynodes.discord import StartTypingNode, StopTypingNode
from edgynodes.discord_llm import BuildChatNode, RespondNode
from edgynodes.discordtmp import ClearTmpDiscordMessagesNode, TemporaryMessageController
import edgynodes as e 
import os
import discord
from discord.ext import commands
import fastmcp
from rich import print as rprint
import asyncio
from typing import Protocol

from mcp_client import get_log_handler, get_progress_handler
from tools import role_dice, leave_voice_channel
from get_llm_node import get_llm_node


logger = setup_logger(__name__)


### STATES

class MyStateProtocol(e.llm.StateProtocol, e.discord.StateProtocol, e.discordtmp.StateProtocol, e.discordmessage.StateProtocol, Protocol):
    pass

class MySharedProtocol(e.llm.SharedProtocol, e.discord.SharedProtocol, e.discordtmp.SharedProtocol, e.discordmessage.SharedProtocol, Protocol):
    pass


class MyState(State):
    llm: e.llm.StateAttribute
    discord: e.discord.StateAttribute
    discordtmp: e.discordtmp.StateAttribute
    discordmessage: e.discordmessage.StateAttribute


class MyShared(Shared):
    llm: e.llm.SharedAttribute
    discord: e.discord.SharedAttribute
    discordtmp: e.discordtmp.SharedAttribute
    discordmessage: e.discordmessage.SharedAttribute


### EDGES

def should_react(shared: MySharedProtocol) -> bool:
    return shared.discordmessage.message.author != shared.discord.bot.user and (   # Prevent reaction on self
        shared.discord.bot.user in shared.discordmessage.message.mentions          # Only when mentioned
        or isinstance(shared.discordmessage.message.channel, discord.DMChannel)    # Or when in DM
    )



### TOOLS


async def join_voice_channel(ctx: ToolContext[MyStateProtocol, MySharedProtocol]) -> str:

    async with ctx.shared.lock:
        bot = ctx.shared.discord.bot
        message = ctx.shared.discordmessage.message

    if not hasattr(message.author, "voice") or not isinstance(message.author.voice.channel, discord.VoiceChannel):
        raise Exception("❌ Du bist in keinem Voice-Channel.")
    
    voice_channel = message.author.voice.channel


    asyncio.create_task(handle_voice(voice_channel, message.channel, bot=bot))

    return f"✅ Erfolgreich **{voice_channel.name}** beigetreten."

### NODES

class DebugNode(Node[MyStateProtocol, MySharedProtocol]):

    async def __call__(self, state: MyStateProtocol, shared: MySharedProtocol) -> None:
        rprint("DEBUG NODE")
        rprint(state)
        rprint(shared)
        rprint("DEBUG NODE END")


### GRAPH HANDLING

async def handle_message(message: discord.Message, bot: commands.Bot) -> None:

            
    # INSTANCES

    temporary_message_controller = TemporaryMessageController(message.channel)

    log_handler = get_log_handler(temporary_message_controller)
    progress_handler = get_progress_handler(temporary_message_controller)

    add_mcp_tools = [
        AddMCPToolsNode(fastmcp.Client(url, log_handler=log_handler, progress_handler=progress_handler)) for url in os.getenv("MCP_SERVER_URLS", "").split(",") if url.strip() != ""
    ]


    state = MyState(
        discordmessage=e.discordmessage.StateAttribute(),
        discordtmp=e.discordtmp.StateAttribute(),
        discord=e.discord.StateAttribute(),
        llm=e.llm.StateAttribute(),
    )
    shared = MyShared(
        discordmessage=e.discordmessage.SharedAttribute(
            message=message
        ),
        discordtmp=e.discordtmp.SharedAttribute(
            controller=temporary_message_controller
        ),
        discord=e.discord.SharedAttribute(
            text_channel=message.channel,
            bot=bot,
        ),
        llm=e.llm.SharedAttribute(),
    )

    llm_node = get_llm_node()
    llm_node_after_max_turns = llm_node.copy()

    build_chat = BuildChatNode(limit=10)
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    add_tools = AddToolsNode([role_dice, join_voice_channel, leave_voice_channel])
    get_new_tool_calls = ExtractNewToolCallsNode()
    respond = RespondNode()
    respond_tool_results = RespondNode()
    respond_after_error = RespondNode()
    notify_max_turns = AddMessageNode(AIMessage(role=AIRoles.USER, chunks=[AIChunkText(text="You have reached the maximum number of turns for this conversation, you cannot call any more tools.")]))
    save_messages = SaveNewMessagesNode()
    save_messages_for_new_turn = SaveNewMessagesNode()
    get_next_tool_call_result = GetNextToolCallResultNode()
    clear_tmp_discord_messages = ClearTmpDiscordMessagesNode()
    integrate_tool_call_results = IntegrateToolResultsNode()
    integrate_mcp_tool_call_results = IntegrateMCPToolResultsNode()
    turn_counter = TurnCounterNode()

    MAX_TURNS = 4

    # GRAPH

    graph = Graph[MyStateProtocol, MySharedProtocol](
        hooks=[NodePrintHook()],
        edges=[
            (
                START,
                lambda st, sh: [start_typing, build_chat, add_tools] if should_react(sh) else None,

                add_tools,
                *add_mcp_tools,
                [llm_node, -turn_counter],
                respond,
                get_new_tool_calls,
                save_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results,

                get_next_tool_call_result,
                clear_tmp_discord_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else integrate_mcp_tool_call_results,

                integrate_mcp_tool_call_results,
                integrate_tool_call_results,
                lambda st, sh: stop_typing if not st.llm.new_messages else respond_tool_results,

                respond_tool_results,
                save_messages_for_new_turn,
                lambda st, sh: llm_node if st.llm.turn_count <= MAX_TURNS else notify_max_turns,

                notify_max_turns,
                llm_node_after_max_turns,
                respond_after_error,

                Exception,
                respond_after_error,
                stop_typing,

                Exception,
                stop_typing,

                END
        )]
    )

    state, shared = await graph(state, shared)

    rprint("Finished handling message.")