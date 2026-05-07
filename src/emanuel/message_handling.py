import discord
from discord.ext import commands
import asyncio
from typing import Protocol

from edgygraph import Graph, START, END, State, Shared
from edgygraph.graph_hooks import NodePrintHook
from edgygraph.nodes import Node
from llmir import AIChunkText, AIMessage, AIRoles
from edgynodes.llm.nodes.tools import ToolContext
from edgynodes.llm.nodes.tools import ExtractNewToolCallsNode, GetNextToolCallResultNode, IntegrateToolResultsNode, IntegrateMCPToolResultsNode, AddToolsNode
from edgynodes.llm.nodes.messages import AddMessageNode, SaveNewMessagesNode
from edgynodes.llm.nodes.turn_counter import IncrementTurnCounterNode
from edgynodes.discord.nodes.typing import StartTypingNode, StopTypingNode
from edgynodes.discord_llm.nodes.build_chat import BuildChatNode
from edgynodes.discord_llm.nodes.respond import RespondNode
from edgynodes.discordtmp import TemporaryMessageController
from edgynodes.discordtmp.nodes.clear import ClearTmpDiscordMessagesNode
from edgynodes.redis.nodes import ReadAllNode, WriteAllNode, CloseNode
import edgynodes as e 

import redis.asyncio as redis

from logger import setup_logger
from tools import role_dice, leave_voice_channel
from get_nodes import get_llm_node, get_mcp_nodes, get_chat_system_message_node
from config import Config

logger = setup_logger(__name__)

voice_available = False

if Config.ENABLE_VOICE:
    from voice_handling import handle_voice
    voice_available = True
elif Config.ENABLE_VOICE is None:
    try:
        from voice_handling import handle_voice
        voice_available = True
    except ImportError:
        logger.warning("Voice handling not available: required optional dependencies not installed")
        voice_available = False


### STATES

class MyStateProtocol(e.llm.StateProtocol, e.discord.StateProtocol, e.discordtmp.StateProtocol, e.discordmessage.StateProtocol, e.redis.StateProtocol, Protocol):
    pass

class MySharedProtocol(e.llm.SharedProtocol, e.discord.SharedProtocol, e.discordtmp.SharedProtocol, e.discordmessage.SharedProtocol, e.redis.SharedProtocol, Protocol):
    pass


class MyState(State):
    llm: e.llm.StateAttribute
    discord: e.discord.StateAttribute
    discordtmp: e.discordtmp.StateAttribute
    discordmessage: e.discordmessage.StateAttribute
    redis: e.redis.StateAttribute

class MyShared(Shared):
    llm: e.llm.SharedAttribute
    discord: e.discord.SharedAttribute
    discordtmp: e.discordtmp.SharedAttribute
    discordmessage: e.discordmessage.SharedAttribute
    redis: e.redis.SharedAttribute


### NODES

class SaveToolCallsToRedisNode(Node[MyStateProtocol, MySharedProtocol]):
    
    async def __call__(self, state: MyStateProtocol, shared: MySharedProtocol) -> None:

        print("Saving new tool calls to Redis: ", shared.llm.new_tool_calls)

        for tool_call in shared.llm.new_tool_calls:
            state.redis.write.append({f"tool_call_{tool_call.id}": tool_call.model_dump_json()})

class SaveToolCallResultsToRedisNode(Node[MyStateProtocol, MySharedProtocol]):

    def __init__(self, max_result_length: int = 1000) -> None:
        super().__init__()
        self.max_result_length = max_result_length


    async def __call__(self, state: MyStateProtocol, shared: MySharedProtocol) -> None:

        print("Saving new tool call results to Redis: ", shared.llm.new_tool_call_results)

        for tool_call, result in shared.llm.new_tool_call_results:
            try:
                string_result = str(result)
                if len(string_result) <= self.max_result_length:  # Limit to prevent excessively large entries
                    state.redis.write.append({f"tool_call_result_{tool_call.id}": string_result})
            except Exception as e:
                logger.error(f"Error occurred while saving tool call result for {tool_call.id}: {e}")


### EDGES

def should_react(shared: MySharedProtocol) -> bool:
    return shared.discordmessage.message.author != shared.discord.bot.user and (   # Prevent reaction on self
        shared.discord.bot.user in shared.discordmessage.message.mentions          # Only when mentioned
        or isinstance(shared.discordmessage.message.channel, discord.DMChannel)    # Or when in DM
    )


### TOOLS


async def join_voice_channel(ctx: ToolContext[MyStateProtocol, MySharedProtocol]) -> str:

    if not voice_available:
        raise Exception("❌ Voice ist nicht verfügbar.")

    async with ctx.shared.lock:
        bot = ctx.shared.discord.bot
        message = ctx.shared.discordmessage.message

    if not hasattr(message.author, "voice") or not isinstance(message.author.voice.channel, discord.VoiceChannel):
        raise Exception("❌ Du bist in keinem Voice-Channel.")
    
    voice_channel = message.author.voice.channel


    asyncio.create_task(handle_voice(voice_channel, message.channel, bot=bot))

    return f"✅ Erfolgreich **{voice_channel.name}** beigetreten."


### GRAPH HANDLING

async def handle_message(message: discord.Message, bot: commands.Bot) -> None:

            
    # INSTANCES

    temporary_message_controller = TemporaryMessageController(message.channel)


    state = MyState(
        discordmessage=e.discordmessage.StateAttribute(),
        discordtmp=e.discordtmp.StateAttribute(),
        discord=e.discord.StateAttribute(),
        llm=e.llm.StateAttribute(),
        redis=e.redis.StateAttribute()
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
        redis=e.redis.SharedAttribute()
    )

    llm_node = get_llm_node()
    llm_node_after_max_turns = llm_node.copy()

    system_message = get_chat_system_message_node()
    build_chat = BuildChatNode(limit=10)
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    add_tools = AddToolsNode([role_dice, join_voice_channel, leave_voice_channel] if voice_available else [role_dice])
    add_mcp_tools = get_mcp_nodes(temporary_message_controller)
    get_new_tool_calls = ExtractNewToolCallsNode()
    save_new_tool_calls_redis = SaveToolCallsToRedisNode()
    save_new_tool_call_results_redis = SaveToolCallResultsToRedisNode()
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
    turn_counter = IncrementTurnCounterNode()
    flush_redis_writes = WriteAllNode()
    close_redis_connection = CloseNode()

    MAX_TURNS = Config.MAX_TURNS

    # GRAPH

    graph = Graph[MyStateProtocol, MySharedProtocol](
        hooks=[NodePrintHook()],
        edges=[
            (
                START,
                lambda st, sh: [start_typing, system_message, add_tools] if should_react(sh) else None,

                system_message,
                build_chat,
                *add_mcp_tools,
                [llm_node, -turn_counter],
                respond,
                get_new_tool_calls,
                save_new_tool_calls_redis,
                save_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else stop_typing,

                get_next_tool_call_result,
                clear_tmp_discord_messages,
                lambda st, sh: get_next_tool_call_result if sh.llm.new_tool_calls else save_new_tool_call_results_redis,

                save_new_tool_call_results_redis,
                lambda st, sh: print(f"Current Redis write queue: {st.redis.write}"),
                
                save_new_tool_call_results_redis,
                flush_redis_writes,
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
                close_redis_connection,

                END
        )]
    )

    state, shared = await graph(state, shared)

    print("Finished handling message.")