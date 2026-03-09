from random import randint

from edgynodes.llm.nodes import tools
from edgynodes.llm import StateProtocol
from edgynodes.discord_llm import SharedProtocol


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


async def leave_voice_channel[T: StateProtocol, S: SharedProtocol](ctx: tools.ToolContext[T, S]) -> str:
    """
    Leave the voice channel.
    """

    async with ctx.shared.lock:
        bot = ctx.shared.discord.bot

    if bot.voice_clients:
        await bot.voice_clients[0].disconnect(force=True)
        return "✅ Erfolgreich Voice-Channel verlassen."

    return "❌ Der Bot ist in keinem Voice-Channel."