"""Compatibility shims for the evaluation dependencies.

ragas (0.4.x) imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at import time only to re-export it; the sunset langchain-community 0.4.x
removed that module. This shim stubs it before ragas is imported anywhere.
"""

import sys
import types


def install() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        shim = types.ModuleType(module_name)

        class ChatVertexAI:  # stub — never used by ragas logic, only re-exported
            pass

        shim.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = shim
