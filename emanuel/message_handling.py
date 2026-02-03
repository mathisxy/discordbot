from llmir import AIMessageToolResponse
from edgygraph import Graph, START, END, Node
from logger import setup_logger
from edgynodes.llm import LLMNodeAzure, LLMNodeOllama, LLMNodeClaude, ExtractNewToolCallsNode, GetNextToolCallResultNode, IntegrateToolResultsNode, IntegrateMCPToolResultsNode, AddToolsNode, SaveNewMessagesNode, LLMNodeGemini, LLMNodeMistral, AddMCPToolsNode, LLMNodeOpenAI
from edgynodes.discord import StartTypingNode, StopTypingNode, DiscordTemporaryMessageController, DiscordState, DiscordShared
from edgynodes.discord_llm import DiscordLLMState, DiscordLLMShared, BuildChatNode, RespondNode
from random import randint
import io
import os
import discord
from discord.ext import commands
import fastmcp
from fastmcp.client.logging import LogMessage
import base64


logger = setup_logger(__name__)


### STATES

class DiscordLLMTmpMsgShared(DiscordLLMShared):
    
    discord_temporary_message_controller: DiscordTemporaryMessageController


### EDGES

def should_react(shared: DiscordLLMShared) -> bool:
    return shared.discord_message.author != shared.discord_bot.user and (   # Prevent reaction on self
        shared.discord_bot.user in shared.discord_message.mentions          # Only when mentioned
        or isinstance(shared.discord_message.channel, discord.DMChannel)    # Or when in DM
    )



### TOOLS

def role_dice(sides: int) -> int:
    """
    Rolls a dice with the given number of sides.
    
    Args:
        sides: Number of sides of the dice.
    
    Returns:
        Result of the dice roll.

    """

    if sides < 2:
        raise Exception

    return randint(1, sides)


async def join_voice_channel(shared: DiscordShared) -> None:
    
    async with shared.lock:
        bot = shared.discord_bot
        message = shared.discord_message

    if not message.author.voice or not message.author.voice.channel:
        raise Exception("❌ Du bist in keinem Voice-Channel.")
    
    voice_channel = message.author.voice.channel

    # Bot ist bereits irgendwo verbunden
    if bot.voice_clients:
        vc = bot.voice_clients[0]

        # Schon im richtigen Channel
        if vc.channel != voice_channel:
            
            # Sonst rüberziehen
            await vc.move_to(voice_channel)

    else:
        # Bot ist noch nirgends verbunden
        await voice_channel.connect()

async def leave_voice_channel(shared: DiscordShared) -> None:

    async with shared.lock:
        bot = shared.discord_bot

    if bot.voice_clients:
        await bot.voice_clients[0].disconnect()


### NODES

class ClearTmpDiscordMessagesNode(Node[DiscordLLMState, DiscordLLMTmpMsgShared]):

    async def run(self, state: DiscordLLMState, shared: DiscordLLMTmpMsgShared) -> None:

        async with shared.lock:

            keys = list(shared.discord_temporary_message_controller.messages.keys())

            for key in keys:
                await shared.discord_temporary_message_controller.delete(key)


### UTILS

class ProgressController:

    @staticmethod
    def create_progress_bar(progress: float, total: float | None, length: int = 20) -> str:
        """Creates a textual progress bar."""
        if total is None or total == 0:
            return f"▰▰▰▰▰ {progress}"
        
        percentage = min(progress / total, 1.0)
        filled = int(length * percentage)
        empty = length - filled
        
        bar = "█" * filled + "░" * empty
        
        return f"{bar}\u2002({progress}/{total})"
    
    @staticmethod
    def get_gradient_color(percentage: float) -> discord.Color:
        """Creates a color that transitions from red to green based on the percentage."""
        percentage = max(0.0, min(1.0, percentage))  # Clamp zwischen 0 und 1
        
        if percentage < 0.5:
            red = 255
            green = int(255 * (percentage * 2))  # 0.0 -> 0.5 becomes 0 -> 255
            blue = 0
        else:
            red = int(255 * (1 - (percentage - 0.5) * 2))  # 0.5 -> 1.0 becomes 255 -> 0
            green = 255
            blue = 0
        
        return discord.Color.from_rgb(red, green, blue)
    
    @classmethod
    async def update_progress(cls, tmp_controller: DiscordTemporaryMessageController, progress: float, total: float | None, key: str = "progress") -> None:
        progress_bar = cls.create_progress_bar(progress, total)
        
        embed = discord.Embed(
            title="Progress",
            description=progress_bar,
            color=discord.Color.blue()
        )
        
        if (msg := tmp_controller.messages.get(key)) is not None and msg.embeds:
            embed = msg.embeds[0]
            embed.description = progress_bar
            
           
        if total is not None:
            percentage = progress / total
            embed.color = cls.get_gradient_color(percentage)
        
        await tmp_controller.update(key=key, embeds=embed)


    @classmethod
    async def update_preview(cls, tmp_controller: DiscordTemporaryMessageController, image: discord.File, key: str = "progress") -> None:

        embed = discord.Embed(
            title="Progress",
        )

        if (msg := tmp_controller.messages.get(key)) is not None and msg.embeds:

            embed = msg.embeds[0]
            embed.set_image(url=f"attachment://{image.filename}")

        await tmp_controller.update(key=key, embeds=embed, files=image)



### GRAPH HANDLING

async def handle_message(message: discord.Message, bot: commands.Bot) -> None:

    temporary_message_controller = DiscordTemporaryMessageController(message.channel)


    # Log Handler for custom MCP Server
    async def log_handler(log_message: LogMessage):
        
        if log_message.data.get("msg") == "preview_image":
            image_base64: str = log_message.data.get("extra").get("base64")
            image_bytes: bytes = base64.b64decode(image_base64)
            image_file: discord.File = discord.File(io.BytesIO(image_bytes), filename="preview.png")

            await ProgressController.update_preview(temporary_message_controller, image_file)


    # Progress Handler
    async def progress_handler(
        progress: float, 
        total: float | None, 
        message: str | None
    ) -> None:
        
        await ProgressController.update_progress(temporary_message_controller, progress, total)

            
    # INSTANCES

    mcp_client = fastmcp.Client("http://localhost:8001/mcp", log_handler=log_handler, progress_handler=progress_handler)

    openai = LLMNodeOpenAI(model="gpt-5.1", api_key=os.getenv("OPENAI_API_KEY", ""), enable_streaming=True)
    claude = LLMNodeClaude(model="claude-haiku-4-5-20251001", api_key=os.getenv("CLAUDE_API_KEY", ""), enable_streaming=True)
    gemini = LLMNodeGemini(model="gemini-3-flash-preview", api_key=os.getenv("GEMINI_API_KEY", ""),enable_streaming=True)
    mistral = LLMNodeMistral(model="mistral-medium-latest", api_key=os.getenv("MISTRAL_API_KEY", ""), enable_streaming=True)
    azure = LLMNodeAzure(model="", api_key=os.getenv("AZURE_API_KEY", ""), base_url=os.getenv("AZURE_BASE_URL", ""), enable_streaming=True)
    ollama = LLMNodeOllama(model="ministral-3:14b", enable_streaming=True)


    state = DiscordLLMState()
    shared = DiscordLLMTmpMsgShared(
        discord_message=message, 
        discord_bot=bot, 
        discord_temporary_message_controller=temporary_message_controller
    )

    llm_node = mistral

    build_chat = BuildChatNode()
    start_typing = StartTypingNode()
    stop_typing = StopTypingNode()
    add_tools = AddToolsNode([role_dice, join_voice_channel, leave_voice_channel])
    add_mcp = AddMCPToolsNode(mcp_client)
    get_new_tool_calls = ExtractNewToolCallsNode()
    get_next_tool_call_result = GetNextToolCallResultNode()
    clear_tmp_discord_messages = ClearTmpDiscordMessagesNode()
    integrate_tool_call_results = IntegrateToolResultsNode()
    integrate_mcp_tool_call_results = IntegrateMCPToolResultsNode()
    respond = RespondNode()
    respond_tool_results = RespondNode()
    save_messages = SaveNewMessagesNode()
    save_messages_for_new_turn = SaveNewMessagesNode()


    # GRAPH

    graph = Graph[DiscordLLMState, DiscordLLMTmpMsgShared](
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
                get_new_tool_calls
            ),
            (
                get_new_tool_calls,
                save_messages
            ),
            (
                save_messages,
                lambda st, sh: get_next_tool_call_result if shared.llm_new_tool_calls else integrate_mcp_tool_call_results
            ),
            (
                get_next_tool_call_result,
                clear_tmp_discord_messages
            ),
            (
                clear_tmp_discord_messages,
                lambda st, sh: get_next_tool_call_result if shared.llm_new_tool_calls else integrate_mcp_tool_call_results
            ),
            (
                integrate_mcp_tool_call_results,
                integrate_tool_call_results
            ),
            (
                integrate_tool_call_results,
                respond_tool_results,
            ),
            (
                respond_tool_results,
                lambda st, sh: stop_typing if not [m for m in st.llm_new_messages if isinstance(m, AIMessageToolResponse)] else save_messages_for_new_turn
            ), 
            (
                save_messages_for_new_turn,
                llm_node
            ),
        ]
    )

    state, shared = await graph(state, shared)

    print("Finished handling message.")