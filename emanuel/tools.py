from random import randint
from typing import Protocol
from edgygraph import StateProtocol

import edgynodes as e # type: ignore


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


class DiscordLLMSharedProtocol(e.llm.SharedProtocol, e.discord.SharedProtocol, Protocol):
    pass


async def leave_voice_channel(ctx: e.llm.ToolContext[StateProtocol, DiscordLLMSharedProtocol]) -> str:
    """
    Leave the voice channel.
    """

    async with ctx.shared.lock:
        bot = ctx.shared.discord.bot

    if bot.voice_clients:
        await bot.voice_clients[0].disconnect(force=True)
        return "✅ Erfolgreich Voice-Channel verlassen."

    return "❌ Der Bot ist in keinem Voice-Channel."