import discord
from discord.ext import commands
from logger import setup_logger
from message_handling import handle_message

from .config import Config


# Logger
logger = setup_logger(__name__)


# Discord Intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.members = True
intents.presences = True
intents.voice_states = True 

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_message(message: discord.Message):
    try:
        await handle_message(message, bot)

    except ExceptionGroup as eg:
        logger.exception(eg)
        first_error = eg.exceptions[0]
        notes = ""
        if hasattr(first_error, "__notes__"):
            notes = "\n".join(first_error.__notes__)
        await message.reply(f"{type(first_error).__name__}: {first_error}\n{notes}")

    except Exception as e:
        logger.exception(e)
        await message.reply(str(e))


@bot.event
async def on_ready():
    print(f"🤖 Bot online as {bot.user}!")


bot.run(Config.DISCORD_TOKEN)