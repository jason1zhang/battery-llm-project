"""
Gradio frontend for Battery LLM.
"""

import os
import gradio as gr
from typing import List

from dotenv import load_dotenv
load_dotenv()


def create_gradio_interface(pipeline):
    """Create a clean Gradio chat interface."""

    def respond(message: str, history: List):
        """Handle user message and return response."""
        if not message.strip():
            return "", history

        try:
            result = pipeline.query(
                question=message,
                return_sources=True
            )

            answer = result.get("answer", "No answer generated")

            # Filter sources
            sources = result.get("sources", [])
            relevant_sources = [
                src for src in sources
                if src.get("similarity") is not None and src.get("similarity") < 1.0
            ]

            # Add sources info at the end of answer
            if relevant_sources:
                source_names = [s.get('metadata', {}).get('source_file', 'Unknown') for s in relevant_sources]
                unique_names = list(dict.fromkeys(source_names))
                answer += f"\n\n**Sources:** {', '.join(unique_names)}"
            elif sources:
                source_name = sources[0].get('metadata', {}).get('source_file', 'Unknown')
                answer += f"\n\n**Source:** {source_name}"

            # Add user and assistant messages to history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return "", history

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history

    # CSS styles
    css = """
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Segoe UI', Roboto, sans-serif !important;
    }
    .gradio-container {
        background: #ffffff !important;
    }
    """

    with gr.Blocks(css=css, title="Battery Assistant") as demo:
        gr.HTML("""
        <div style="max-width: 800px; margin: 0 auto; padding: 20px; text-align: center;">
            <h1 style="font-size: 20px; font-weight: 600; color: #1f2937;">Battery Manufacturing Assistant</h1>
            <p style="color: #6b7280; font-size: 14px;">Ask about battery manufacturing processes, quality control, and safety</p>
        </div>
        """)

        chatbot = gr.Chatbot(
            height=500,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask about battery manufacturing...",
                scale=5,
            )
            submit_btn = gr.Button("Send", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.HTML("""
        <div style="max-width: 800px; margin: 20px auto; padding: 0 20px;">
            <p style="font-size: 12px; color: #9ca3af;">MiniMax-M2.7 · 384-dim · RAG</p>
        </div>
        """)

        # Event handlers
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        clear_btn.click(
            fn=lambda: ("", []),
            outputs=[msg, chatbot],
        )

    return demo


def run_gradio(pipeline, port=7860):
    """Run the Gradio interface."""
    demo = create_gradio_interface(pipeline)
    demo.launch(
        server_port=port,
        server_name="0.0.0.0",
        share=False,
    )


if __name__ == "__main__":
    from src.rag.pipeline import create_pipeline

    print("Initializing RAG pipeline...")
    pipeline = create_pipeline(
        data_path="data/raw",
        persist_directory="data/embeddings/chroma"
    )

    print("Starting Gradio interface...")
    run_gradio(pipeline)
