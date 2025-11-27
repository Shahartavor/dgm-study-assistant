import gradio as gr
from dgm_study_assistant.llm.provider import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dgm_study_assistant.rag.loader import build_rag_chain

llm = get_llm()
rag_chain = build_rag_chain(llm)
def answer(query):
    return rag_chain.invoke({"question": query})


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


query_box = gr.Textbox(
    label="query",
    lines=2,          # how tall the input box is
    placeholder="Ask about VAEs, GANs, Diffusion, EM, GMMs..."
)

output_box = gr.Textbox(
    label="output",
    lines=12,         # make this big so you can see the whole answer
)

gr.Interface(
    fn=answer,
    inputs=query_box,
    outputs=output_box,
    title="DGM Study Assistant",
).launch()