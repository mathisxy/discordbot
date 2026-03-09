from collections.abc import Sequence

from edgynodes.llm.core.nodes import LLMNode # type: ignore
from edgynodes.discordtmp import TemporaryMessageController # type: ignore
from edgynodes.llm.nodes.tools.core import AddMCPToolsNode # type: ignore

from config import Config

def get_llm_node() -> LLMNode:
    
    key = Config.LLM_PROVIDER.lower()
    streaming = Config.LLM_STREAMING.lower() == "true"

    match key:
        case "openai":
            from edgynodes.llm.nodes.openai import LLMOpenAINode # type: ignore

            model = Config.OPENAI_MODEL
            api_key = Config.OPENAI_API_KEY
            return LLMOpenAINode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "claude":
            from edgynodes.llm.nodes.openai import LLMClaudeNode # type: ignore

            model = Config.CLAUDE_MODEL
            api_key = Config.CLAUDE_API_KEY
            return LLMClaudeNode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "gemini":
            from edgynodes.llm.nodes.openai import LLMGeminiNode # type: ignore

            model = Config.GEMINI_MODEL
            api_key = Config.GEMINI_API_KEY
            return LLMGeminiNode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "mistral":
            from edgynodes.llm.nodes.openai import LLMMistralNode # type: ignore

            model = Config.MISTRAL_MODEL
            api_key = Config.MISTRAL_API_KEY
            return LLMMistralNode(
                model=model,
                api_key=api_key,
                stream=streaming
            )
        case "azure":
            from edgynodes.llm.nodes.openai import LLMAzureNode # type: ignore

            model = Config.AZURE_MODEL
            api_key = Config.AZURE_API_KEY
            base_url = Config.AZURE_BASE_URL
            return LLMAzureNode(
                model=model,
                api_key=api_key,
                base_url=base_url,
                enable_streaming=streaming
            )
        case "ollama":
            from edgynodes.llm.nodes.openai import LLMOllamaNode # type: ignore
            model = Config.OLLAMA_MODEL
            base_url = Config.OLLAMA_BASE_URL
            return LLMOllamaNode(
                model=model,
                *{base_url: base_url} if base_url else {},
                enable_streaming=streaming,
                keep_alive="0s"
            )
        case _:
            raise ValueError(f"Unsupported LLM provider: {key}")
        

def get_mcp_nodes(temporary_message_controller: TemporaryMessageController) -> Sequence[AddMCPToolsNode]:

    from edgynodes.llm.nodes.tools.core import AddMCPToolsNode # type: ignore
    from fastmcp import Client
    from mcp_client import get_log_handler, get_progress_handler

    nodes: list[AddMCPToolsNode] = []

    log_handler = get_log_handler(temporary_message_controller)
    progress_handler = get_progress_handler(temporary_message_controller)


    for url in Config.MCP_SERVER_URLS:

        nodes.append(AddMCPToolsNode(Client(url, log_handler=log_handler, progress_handler=progress_handler)))

    return nodes

