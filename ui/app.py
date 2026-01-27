import gradio as gr
import random
from dgm_study_assistant.llm.provider import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dgm_study_assistant.rag.loader import build_rag_chain

from dgm_study_assistant.rag.loader import get_slide_image_from_metadata

SYSTEM_MESSAGE = SystemMessage(content="""
You are a helpful assistant specialized in Deep Generative Models (DGM).
Focus on topics such as VAEs, GANs, Diffusion Models, Normalizing Flows, EM algorithm, GMMs, latent variable models, and score-based methods.

====================================================================
STYLE RULES
====================================================================
• Respond in clear, structured English.
• Be concise but mathematically precise.
• All mathematical expressions MUST be written as LaTeX **display equations** using:

$$
... LaTeX ...
$$

• Never output LaTeX inside square brackets [ ].
• Never output inline raw LaTeX without $$ delimiters.
• Prefer proper mathematical notation over long verbal descriptions.

====================================================================
STRICT RAG GROUNDING RULES (CRITICAL)
====================================================================
Before answering, you MUST check whether the user's question is covered by the transcript/context provided by the RAG system.

You MUST follow one of these two response formats below:

1. If the answer **is supported by the transcript/context**, then begin your answer with:
   "### From transcript:"
   and base your explanation ONLY on the transcript/context.

2. If the answer is **NOT supported by the transcript/context**, then begin your answer with:
   "### Not in transcript — general knowledge:"
   and then give a correct explanation from your general ML understanding.

ADDITIONAL RULES:
• You MUST include one of these two headers in every answer. No exceptions.
• Never claim the transcript contains material that it does not.
• Never invent equations, variables, symbols, or derivations not present in the transcript unless clearly marked as general knowledge.
• If you are uncertain, choose the most conservative and context-aligned interpretation.

====================================================================
CONVERSATION BEHAVIOR
====================================================================
• Be patient and encouraging.
• If the user provides incorrect or malformed math, gently correct it.
• Use step-by-step derivations when helpful.
• For multi-step math, use multiple display equation blocks.

Your highest priorities are: accuracy, grounding, mathematical rigor, and clear communication.
""")



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
    
    # Add message in the new format
    history.append({"role": "user", "content": message})
    return history, ""


def get_bot_response(history, evaluate_answer=None):
    # If there's no new user message waiting for an answer, do nothing
    if not history or history[-1].get("role") != "user":
        # Also make sure slide and source are hidden in this "no-op" case
        return history, gr.update(visible=False, value=None), gr.update(visible=False, value="")

    user_message = history[-1]["content"]

    # Run the full RAG chain
    result = rag_chain.invoke({"question": user_message})

    answer = result["answer"]  # LLM output
    docs = result["docs"]      # retrieved documents

    # Debug prints (optional)
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:200])
        print("-------------------------")

    # Append assistant message to chat history
    final_answer = answer
    if evaluate_answer:
        # Import evaluation module only when needed to avoid nest_asyncio issues
        from dgm_study_assistant.evaluation import RAGEvaluator
        from dgm_study_assistant.rag.loader import get_embeddings
        
        evaluator = RAGEvaluator(
            judge_llm=llm,
            embedding_model=get_embeddings(),
            rag_chain=rag_chain,
        )

        eval_result = evaluator.evaluate_single_sample(user_message)

        if eval_result:
            evaluation_lines = ["\n\n---\n### 🧪 Evaluation"]
            
            # Helper function to format metric values
            def format_metric(value, name):
                if value is not None:
                    return f"- **{name}**: {value:.3f}"
                else:
                    return f"- **{name}**: *Unable to calculate*"
            
            faith = eval_result.get("faithfulness")
            relev = eval_result.get("answer_relevancy")
            recall = eval_result.get("context_recall")
            precision = eval_result.get("context_precision")

            evaluation_lines.append(format_metric(recall, "Context Recall"))
            evaluation_lines.append(format_metric(precision, "Context Precision"))
            evaluation_lines.append(format_metric(faith, "Faithfulness"))
            evaluation_lines.append(format_metric(relev, "Answer Relevancy"))
            
            # Add explanation for unavailable metrics
            unavailable_count = sum([recall is None, precision is None, faith is None, relev is None])
            if unavailable_count > 0:
                evaluation_lines.append("\n*Some metrics couldn't be calculated. This may be due to:")
                evaluation_lines.append("  - Insufficient context information")
                evaluation_lines.append("  - LLM processing limitations")
                evaluation_lines.append("  - Missing reference data for comparison*")

            evaluation_text = "\n".join(evaluation_lines)
            final_answer += evaluation_text
    history.append({"role": "assistant", "content": final_answer})

    # Try to find a slide image and its source
    slide_path = None
    slide_source_text = ""
    
    if docs:
        source_info = []
        unique_sources = set()
        
        for doc in docs:
            metadata = doc.metadata
            if metadata.get("source") == "slides":
                pdf_stem = metadata.get("pdf_stem", "Unknown")
                page = metadata.get("page", "Unknown")
                source_key = f"{pdf_stem}"
                
                if source_key not in unique_sources:
                    unique_sources.add(source_key)
                    # Format the pdf_stem to be more readable
                    formatted_name = pdf_stem.replace("_", " ").title()
                    source_info.append(f"📄 **{formatted_name}** (Page {page})")
            
            # Get slide image
            if not slide_path:
                slide_path = get_slide_image_from_metadata(doc.metadata)
        
        if source_info:
            slide_source_text = "### Source\n" + "\n".join(source_info)
    
    if slide_path:
        # Show the image + set value
        slide_update = gr.update(visible=True, value=slide_path)
        source_update = gr.update(visible=True, value=slide_source_text)
    else:
        # Hide the components
        slide_update = gr.update(visible=False, value=None)
        source_update = gr.update(visible=False, value="")

    return history, slide_update, source_update


def clear_chat():
    """Clear chat history and reset input box."""
    return [], "", gr.update(visible=False, value=None), gr.update(visible=False, value="")


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
                    render_markdown=True,
                    type='messages',
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False}
                    ],
                    show_share_button=False
                )
                slide_viewer = gr.Image(
                    label="Relevant Slide",
                    visible=False,
                    height=350,
                    interactive=False
                )
                slide_source = gr.Markdown(
                    visible=False,
                    elem_id="slide_source"
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
                evaluate_toggle = gr.Checkbox(
                    label="🧪 Evaluate Answer",
                    value=False
                )
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
        _setup_event_handlers(submit_btn, query_box, chatbot, slide_viewer, slide_source, clear_btn,
                             recommendation_buttons, refresh_btn, evaluate_toggle)
    
    return demo

def _setup_event_handlers(submit_btn, query_box, chatbot, slide_viewer, slide_source, clear_btn,
                         recommendation_buttons, refresh_btn, evaluate_toggle):
    """Configure all event handlers for the interface."""
    # Submit handlers (both button and Enter key)
    submit_btn.click(
        fn=add_user_message,
        inputs=[query_box, chatbot],
        outputs=[chatbot, query_box]
    ).then(
        fn=get_bot_response,
        inputs=[chatbot, evaluate_toggle],
        outputs=[chatbot, slide_viewer, slide_source]
    )
    
    query_box.submit(
        fn=add_user_message,
        inputs=[query_box, chatbot],
        outputs=[chatbot, query_box]
    ).then(
        fn=get_bot_response,
        inputs=[chatbot, evaluate_toggle],
        outputs=[chatbot, slide_viewer, slide_source]
    )
    
    # Clear chat handler
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, query_box, slide_viewer, slide_source]
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