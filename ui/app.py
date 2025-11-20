import gradio as gr
from src.llm.provider import get_llm

llm = get_llm()

def chat_fn(message, history):
    response = llm.invoke(message)
    history.append((message, response.content))
    return history, history


with gr.Blocks() as demo:
    gr.Markdown("# **DGM Study Assistant (Simple Chatbot)**")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything about EM, GMM, VAEs...")

    def user_submit(user_message, chat_history):
        return "", chat_history + [[user_message, None]]

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        chat_fn, [msg, chatbot], [chatbot, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
