import gradio as gr
import random
from dgm_study_assistant.llm.provider import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dgm_study_assistant.rag.loader import build_rag_chain

SYSTEM_MESSAGE = SystemMessage(content="""
You are a helpful assistant specialized in Deep Generative Models (DGM).
Focus on topics like VAEs, GANs, Diffusion Models, Normalizing Flows, EM algorithm, GMMs, etc.
Respond in clear, structured English with examples and LaTeX math when helpful.
Be patient and encouraging.
""".strip())

RECOMMENDED_QUESTIONS = [
    "What are Variational Autoencoders (VAEs) and how do they work?",
    "Explain the difference between GANs and VAEs",
    "How does the reparameterization trick work in VAEs?",
    "What is the ELBO (Evidence Lower Bound) in VAEs?",
    "How do diffusion models generate images?",
    "What are normalizing flows and their applications?",
    "Explain the EM algorithm with examples",
    "How do Gaussian Mixture Models work?",
    "What is the mode collapse problem in GANs?",
    "How does the denoising process work in diffusion models?",
    "What are the advantages of score-based generative models?",
    "Explain the concept of latent space in generative models",
    "How do you train a GAN effectively?",
    "What is the difference between conditional and unconditional generation?",
    "How do you evaluate generative models?"
]

llm = get_llm()
rag_chain = build_rag_chain(llm)

def add_user_message(message, history):
    """Add user message to chat history immediately."""
    if not message.strip():
        return history, ""
    
    history.append([message, None])
    return history, ""

def get_bot_response(history):
    """AI response with chat history context and RAG integration."""
    if not history or history[-1][1] is not None:
        return history
    
    user_message = history[-1][0]
    
    # First get RAG context for the current question
    rag_context = rag_chain.invoke({"question": user_message})
    
    # Build conversation context from history
    messages = [SYSTEM_MESSAGE]
    
    # Add previous conversation turns (excluding the current incomplete one)
    for user_msg, bot_msg in history[:-1]:
        if user_msg and bot_msg:  # Only add complete exchanges
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
    
    # Always add the current user message to maintain conversation flow
    messages.append(HumanMessage(content=user_message))
    
    # Add RAG context as additional context, not as replacement
    if rag_context and rag_context.strip():
        context_message = HumanMessage(content=f"""Additional context from course materials that might be relevant:

{rag_context}

Please use this context if relevant to answer the question, but prioritize our conversation history for understanding what the user is asking about.""")
        messages.append(context_message)
    
    # Generate response
    full_response = ""
    for chunk in llm.stream(messages):
        full_response += chunk.content
    
    history[-1][1] = full_response
    return history

def clear_chat():
    """Clear chat history and reset input box."""
    return [], ""

def get_random_recommendations(num=5):
    """Get (random) sample of recommended questions."""
    return random.sample(RECOMMENDED_QUESTIONS, min(num, len(RECOMMENDED_QUESTIONS)))

def use_recommended_question(question):
    """Fill the input box with selected recommended question."""
    return question

def refresh_recommendations():
    """Generate new set of random recommendations."""
    new_questions = get_random_recommendations(6)
    return [gr.update(value=q) for q in new_questions]

def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="DGM Study Assistant") as demo:
        # Header
        gr.Markdown("# 🤖 Deep Generative Models Study Assistant")
        gr.Markdown("Ask questions about VAEs, GANs, Diffusion Models, Normalizing Flows, EM algorithm, GMMs, and more!")
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    show_copy_button=True,
                    show_copy_all_button=False,
                    bubble_full_width=False,
                    avatar_images=(
                        "https://api.dicebear.com/7.x/thumbs/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=bot"
                    ),
                    show_share_button=False
                )
                
                # Input area
                with gr.Row():
                    query_box = gr.Textbox(
                        label="",
                        placeholder="Ask about VAEs, GANs, Diffusion, EM, GMMs...",
                        scale=4,
                        lines=1,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
            
            # Recommendations sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Recommended Questions")
                gr.Markdown("*Click any question to use it:*")
                
                # Generate recommendation buttons
                initial_recommendations = get_random_recommendations(6)
                recommendation_buttons = []
                
                for question in initial_recommendations:
                    btn = gr.Button(
                        question,
                        size="sm",
                        variant="secondary",
                        elem_classes=["recommendation-btn"]
                    )
                    recommendation_buttons.append(btn)
                
                refresh_btn = gr.Button("🔄 New Questions", size="sm", variant="outline")
        
        # Wire up event handlers
        _setup_event_handlers(submit_btn, query_box, chatbot, clear_btn, 
                             recommendation_buttons, refresh_btn)
    
    return demo

def _setup_event_handlers(submit_btn, query_box, chatbot, clear_btn, 
                         recommendation_buttons, refresh_btn):
    """Configure all event handlers for the interface."""
    # Submit handlers (both button and Enter key)
    submit_btn.click(
        fn=add_user_message,
        inputs=[query_box, chatbot],
        outputs=[chatbot, query_box]
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    query_box.submit(
        fn=add_user_message,
        inputs=[query_box, chatbot],
        outputs=[chatbot, query_box]
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    # Clear chat handler
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, query_box]
    )
    
    # Recommendation button handlers
    for btn in recommendation_buttons:
        btn.click(
            fn=use_recommended_question,
            inputs=[btn],
            outputs=[query_box]
        )
    
    # Refresh recommendations handler
    refresh_btn.click(
        fn=refresh_recommendations,
        outputs=recommendation_buttons
    )

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()