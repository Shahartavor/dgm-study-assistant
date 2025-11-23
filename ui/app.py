import gradio as gr
from dgm_study_assistant.llm.provider import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = get_llm()

_SYSTEM_MESSAGE = SystemMessage(content="""
You are a helpful assistant specialized in Deep Generative Models (DGM).
Focus on topics like VAEs, GANs, Diffusion Models, Normalizing Flows, EM algorithm, GMMs, etc.
Respond in clear, structured English with examples and LaTeX math when helpful.
Be patient and encouraging.
""".strip())


def respond(message: str, history: list):
    messages = [_SYSTEM_MESSAGE]
    # Add previous conversation turns
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # Add current user message
    messages.append(HumanMessage(content=message))

    full_response = ""
    for chunk in llm.stream(messages):
        full_response += chunk.content
        yield full_response



gr.ChatInterface(
    fn=respond,
    type="messages",
    title="DGM Study Assistant",
    description="Expert tutor for Deep Generative Models — VAEs, GANs, Diffusion, Flows, EM & more. Clear answers with examples and math.",
    chatbot=gr.Chatbot(type="messages", height=300),
    textbox=gr.Textbox(placeholder="Ask me anything about VAEs, GANs, Diffusion Models, EM algorithm...", label="Your question"),
    submit_btn="Send",
    autofocus=True,
).launch()
