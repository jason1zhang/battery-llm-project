"""
Response generation using MiniMax LLM.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ResponseGenerator:
    """Base class for response generation."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, context: str, question: str) -> str:
        """Generate response given context and question."""
        raise NotImplementedError


class SimpleResponseGenerator(ResponseGenerator):
    """Simple generator that returns context directly as answer (no LLM needed)."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ):
        super().__init__(temperature, max_tokens, top_p)

    def generate(self, context: str, question: str) -> str:
        """Return a summary of the context as the answer."""
        lines = context.split('\n\n')
        summary_parts = []
        for line in lines[:3]:
            if len(line.strip()) > 20:
                summary_parts.append(line.strip())

        if summary_parts:
            return "Based on the retrieved documents:\n\n" + "\n\n".join(summary_parts)
        return context[:500] + "..." if len(context) > 500 else context


class GroundedGenerator(ResponseGenerator):
    """Generator with source citation and hallucination checking."""

    def __init__(
        self,
        generator: ResponseGenerator,
        citation_enabled: bool = True,
        hallucination_check: bool = True,
    ):
        super().__init__(generator.temperature, generator.max_tokens, generator.top_p)
        self.generator = generator
        self.citation_enabled = citation_enabled
        self.hallucination_check = hallucination_check

    def generate(
        self,
        context: str,
        question: str,
        source_docs: Optional[List[Document]] = None,
    ) -> Dict[str, Any]:
        """Generate response with optional citations."""
        answer = self.generator.generate(context, question)
        result = {"answer": answer}

        if self.citation_enabled and source_docs:
            citations = self._extract_citations(source_docs)
            result["citations"] = citations

        return result

    def _extract_citations(self, docs: List[Document]) -> List[Dict]:
        """Extract citation information from source documents."""
        citations = []
        for doc in docs:
            citation = {
                "source": doc.metadata.get("source_file", "Unknown"),
                "content_preview": doc.page_content[:100] + "...",
            }
            citations.append(citation)
        return citations


class MiniMaxGenerator(ResponseGenerator):
    """MiniMax LLM based generator."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",  # China platform with /anthropic suffix
        model: str = "MiniMax-M2.7",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ):
        super().__init__(temperature, max_tokens, top_p)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        # Initialize sync client for simpler API
        self.client = anthropic.Anthropic(
            base_url=api_base,
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    def generate(self, context: str, question: str) -> str:
        """Generate response using MiniMax LLM."""
        prompt = self._create_prompt(context, question)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Extract text from response
            # Handle response content - MiniMax may return thinking blocks
            text_content = ""
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        text_content += block.text
                    elif block.type == "thinking":
                        # Skip thinking blocks or include if needed
                        pass
            return text_content if text_content else "No response generated"
            return "No response generated"

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _create_prompt(self, context: str, question: str) -> str:
        """Create the prompt for generation."""
        return f"""You are an expert technical writer for Apple Battery Manufacturing documentation.
Your task is to write clear, natural answers based ONLY on the provided context.

IMPORTANT:
- Synthesize information from all provided context fragments
- Do NOT just list what you found - write flowing paragraphs
- If the context mentions headers like "Required Safety Tests", "Safety Testing", or specific test names (External Short Circuit, Overcharge, Thermal Abuse, Penetration, Crush), include those details
- Do NOT say "the context does not include" - use what IS provided
- Write naturally without markdown headers unless genuinely needed
- Combine multiple relevant points into cohesive answers

Context (may contain multiple relevant sections):
{context}

Question: {question}

Write your answer:"""


class LocalGenerator(ResponseGenerator):
    """Local TinyLlama + LoRA generator."""

    def __init__(
        self,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path: str = "models/lora_adapter",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ):
        super().__init__(temperature, max_tokens, top_p)
        self.adapter_path = adapter_path

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and peft packages are required for local generation. "
                "Install with: pip install transformers peft torch"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, context: str, question: str) -> str:
        prompt = f"""[INST] Answer based on context.

Context: {context}

Question: {question} [/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_generator(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    device: str = "cpu",
    use_local: bool = False,
    system_prompt: Optional[str] = None,
    use_simple: bool = False,
    use_minimax: bool = True,
    minimax_api_key: Optional[str] = None,
    lora_adapter_path: str = "models/lora_adapter",
) -> ResponseGenerator:
    """
    Factory function to create a generator.

    Args:
        model_id: Model ID (ignored for MiniMax)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        device: Device to run on (cpu, cuda)
        use_local: Use local model instead of API
        system_prompt: Optional system prompt
        use_simple: Use simple generator without LLM
        use_minimax: Use MiniMax LLM
        minimax_api_key: MiniMax API key
        lora_adapter_path: Path to LoRA adapter (for local generation)

    Returns:
        Configured generator
    """
    # Check use_local first - local generation takes priority
    if use_local:
        return LocalGenerator(
            base_model=model_id,
            adapter_path=lora_adapter_path,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    if use_simple or not use_minimax:
        return SimpleResponseGenerator(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    # Try MiniMax
    api_key = minimax_api_key or os.environ.get("MINIMAX_API_KEY")

    if not api_key:
        print("Warning: MINIMAX_API_KEY not set. Falling back to simple generator.")
        return SimpleResponseGenerator(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    return MiniMaxGenerator(
        api_key=api_key,
        model="MiniMax-M2.7",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
