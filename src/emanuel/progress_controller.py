from edgynodes.discordtmp import TemporaryMessageController # type: ignore
import discord

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
    async def update_progress(cls, tmp_controller: TemporaryMessageController, progress: float, total: float | None, key: str = "progress") -> None:
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
    async def update_preview(cls, tmp_controller: TemporaryMessageController, image: discord.File, key: str = "progress") -> None:

        embed = discord.Embed(
            title=key.capitalize(),
        )

        if (msg := tmp_controller.messages.get(key)) is not None and msg.embeds:

            embed = msg.embeds[0]
            embed.set_image(url=f"attachment://{image.filename}")

        await tmp_controller.update(key=key, embeds=embed, files=image)


