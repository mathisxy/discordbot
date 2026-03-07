from edgygraph import Node
import os

from edgynodes.llm import StateProtocol, SharedProtocol # type: ignore

def get_llm_node() -> Node[StateProtocol, SharedProtocol]:
    
    key = os.getenv("LLM_PROVIDER", "").lower()
    streaming = os.getenv("LLM_STREAMING", "").lower() == "true"

    match key:
        case "openai":
            from edgynodes.llm.nodes.openai import LLMOpenAINode # type: ignore

            model = os.getenv("OPENAI_MODEL", "")
            api_key = os.getenv("OPENAI_API_KEY", "")
            return LLMOpenAINode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "claude":
            from edgynodes.llm.nodes.openai import LLMClaudeNode # type: ignore

            model = os.getenv("CLAUDE_MODEL", "")
            api_key = os.getenv("CLAUDE_API_KEY", "")
            return LLMClaudeNode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "gemini":
            from edgynodes.llm.nodes.openai import LLMGeminiNode # type: ignore

            model = os.getenv("GEMINI_MODEL", "")
            api_key = os.getenv("GEMINI_API_KEY", "")
            return LLMGeminiNode(
                model=model,
                api_key=api_key,
                enable_streaming=streaming
            )
        case "mistral":
            from edgynodes.llm.nodes.openai import LLMMistralNode # type: ignore

            model = os.getenv("MISTRAL_MODEL", "")
            api_key = os.getenv("MISTRAL_API_KEY", "")
            return LLMMistralNode(
                model=model,
                api_key=api_key,
                stream=streaming
            )
        case "azure":
            from edgynodes.llm.nodes.openai import LLMAzureNode # type: ignore

            model = os.getenv("AZURE_MODEL", "")
            api_key = os.getenv("AZURE_API_KEY", "")
            base_url = os.getenv("AZURE_BASE_URL", "")
            return LLMAzureNode(
                model=model,
                api_key=api_key,
                base_url=base_url,
                enable_streaming=streaming
            )
        case "ollama":
            from edgynodes.llm.nodes.openai import LLMOllamaNode # type: ignore
            model = os.getenv("OLLAMA_MODEL", "")
            return LLMOllamaNode(
                model=model,
                enable_streaming=streaming,
                keep_alive="0s"
            )
        case _:
            raise ValueError(f"Unsupported LLM provider: {key}")