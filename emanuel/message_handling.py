from edgygraph import Graph, START, END, Node
from voice_handling import handle_voice
from logger import setup_logger
from edgynodes.llm import LLMAzureNode, LLMOllamaNode, LLMClaudeNode, ExtractNewToolCallsNode, GetNextToolCallResultNode, IntegrateToolResultsNode, IntegrateMCPToolResultsNode, AddToolsNode, SaveNewMessagesNode, LLMGeminiNode, LLMMistralNode, AddMCPToolsNode, LLMOpenAINode
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

from mcp_client import get_log_handler, get_progress_handler
from tools import role_dice, leave_voice_channel


logger = setup_logger(__name__)


### STATES

class MyState(e.discordmessage.State, e.discord_discordmessage.State, e.discord_llm.State, e.discord.State, e.llm.State, e.discordtmp.State):
    pass

class MyShared(e.discordmessage.Shared, e.discord_discordmessage.Shared, e.discord_llm.Shared, e.discord.Shared, e.llm.Shared, e.discordtmp.Shared):
    pass

### EDGES

def should_react(shared: e.discord_discordmessage.Shared) -> bool:
    return shared.discordmessage.message.author != shared.discord.bot.user and (   # Prevent reaction on self
        shared.discord.bot.user in shared.discordmessage.message.mentions          # Only when mentioned
        or isinstance(shared.discordmessage.message.channel, discord.DMChannel)    # Or when in DM
    )



### TOOLS


async def join_voice_channel(shared: e.discord_discordmessage.Shared) -> str:

    async with shared.lock:
        bot = shared.discord.bot
        message = shared.discordmessage.message

    if not hasattr(message.author, "voice") or not isinstance(message.author.voice.channel, discord.VoiceChannel):
        raise Exception("❌ Du bist in keinem Voice-Channel.")
    
    voice_channel = message.author.voice.channel


    asyncio.create_task(handle_voice(voice_channel, message.channel, bot=bot))

    return f"✅ Erfolgreich **{voice_channel.name}** beigetreten."

### NODES

class DebugNode(Node[MyState, MyShared]):

    async def run(self, state: MyState, shared: MyShared) -> None:
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

    mcp_client = fastmcp.Client("http://localhost:8001/mcp", log_handler=log_handler, progress_handler=progress_handler)

    openai = LLMOpenAINode(model="gpt-5.1", api_key=os.getenv("OPENAI_API_KEY", ""), enable_streaming=True)
    claude = LLMClaudeNode(model="claude-haiku-4-5-20251001", api_key=os.getenv("CLAUDE_API_KEY", ""), enable_streaming=True)
    gemini = LLMGeminiNode(model="gemini-3-flash-preview", api_key=os.getenv("GEMINI_API_KEY", ""),enable_streaming=True)
    mistral = LLMMistralNode(model="mistral-medium-latest", api_key=os.getenv("MISTRAL_API_KEY", ""), stream=True)
    azure = LLMAzureNode(model="", api_key=os.getenv("AZURE_API_KEY", ""), base_url=os.getenv("AZURE_BASE_URL", ""), enable_streaming=True)
    ollama = LLMOllamaNode(model="ministral-3:14b", enable_streaming=True)

    debug_node = DebugNode()


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

    llm_node = mistral

    build_chat = BuildChatNode()
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    add_tools = AddToolsNode([role_dice, join_voice_channel, leave_voice_channel])
    add_mcp = AddMCPToolsNode(mcp_client)
    get_new_tool_calls = ExtractNewToolCallsNode()
    respond = RespondNode()
    respond_tool_results = RespondNode()
    save_messages = SaveNewMessagesNode()
    save_messages_for_new_turn = SaveNewMessagesNode()
    get_next_tool_call_result = GetNextToolCallResultNode()
    clear_tmp_discord_messages = ClearTmpDiscordMessagesNode()
    integrate_tool_call_results = IntegrateToolResultsNode()
    integrate_mcp_tool_call_results = IntegrateMCPToolResultsNode()

    # GRAPH

    graph = Graph[MyState, MyShared](
        edges=[
            (
                START,
                lambda st, sh: start_typing if should_react(sh) else END
            ),
            (
                start_typing,
                build_chat
            ),
            (
                build_chat,
                add_tools
            ),
            (
                add_tools,
                add_mcp
            ),
            (
                add_mcp,
                llm_node
            ),
            (
                llm_node,
                respond
            ),
            (
                respond,
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
                lambda st, sh: stop_typing if not st.llm.new_messages else respond_tool_results
            ), 
            (
                respond_tool_results,
                save_messages_for_new_turn,
            ),
            (
                save_messages_for_new_turn,
                llm_node
            ),
        ]
    )

    state, shared = await graph(state, shared)

    rprint("Finished handling message.")