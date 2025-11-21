import gradio as gr
from dgm_study_assistant.llm.provider import get_llm

llm = get_llm()

def chat_fn(message, history):
    resp = llm.invoke(message)
    history = history + [(message, resp.content)]
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# DGM Study Assistant")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask about EM, GMM, VAEs...")

    def user_submit(user_message, chat_history):
        return "", chat_history + [[user_message, None]]

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        chat_fn, [msg, chatbot], [chatbot, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
