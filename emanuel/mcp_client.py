from edgynodes.discord import TemporaryMessageController # type: ignore 
from fastmcp.client.logging import LogMessage
import discord
import io
import base64

from progress_controller import ProgressController


def get_log_handler(temporary_message_controller: TemporaryMessageController):


    # Log Handler for custom MCP Server
    async def log_handler(log_message: LogMessage):
        
        if log_message.data.get("msg") == "preview_image":
            image_base64: str = log_message.data.get("extra").get("base64")
            image_bytes: bytes = base64.b64decode(image_base64)
            image_file: discord.File = discord.File(io.BytesIO(image_bytes), filename="preview.png")

            await ProgressController.update_preview(temporary_message_controller, image_file)

    return log_handler

def get_progress_handler(temporary_message_controller: TemporaryMessageController):


    # Progress Handler
    async def progress_handler(
        progress: float, 
        total: float | None, 
        message: str | None
    ) -> None:
        
        await ProgressController.update_progress(temporary_message_controller, progress, total)

    return progress_handler