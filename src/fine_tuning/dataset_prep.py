"""
Dataset preparation for fine-tuning.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document


class InstructionDatasetFormatter:
    """Format data for instruction tuning."""

    @staticmethod
    def format_alpaca(
        instruction: str,
        input_text: str = "",
        output: str = "",
    ) -> Dict[str, str]:
        """
        Format as Alpaca instruction format.

        Args:
            instruction: The instruction
            input_text: Optional input context
            output: The expected output

        Returns:
            Formatted dictionary
        """
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

    @staticmethod
    def format_sharegpt(
        conversations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Format as ShareGPT format."""
        return {"conversations": conversations}


class DomainDatasetPreparer:
    """Prepare domain-specific datasets for fine-tuning."""

    def __init__(self):
        self.examples = []

    def add_example(
        self,
        instruction: str,
        input_text: str,
        output: str,
    ):
        """Add a training example."""
        self.examples.append(
            InstructionDatasetFormatter.format_alpaca(
                instruction=instruction,
                input_text=input_text,
                output=output,
            )
        )

    def create_qa_pairs_from_documents(
        self,
        documents: List[Document],
    ) -> List[Dict[str, str]]:
        """Create QA pairs from documents."""
        qa_pairs = []

        for doc in documents:
            content = doc.page_content
            source = doc.metadata.get("source_file", "Unknown")

            # Extract key information for Q&A
            # This is a simplified version - in production, use more sophisticated methods

            # Split content into sections
            sections = content.split("\n\n")

            for i, section in enumerate(sections):
                if len(section) < 50:
                    continue

                # Create questions based on section content
                questions = self._generate_questions_from_content(section)

                for question in questions:
                    qa_pairs.append(
                        InstructionDatasetFormatter.format_alpaca(
                            instruction=question,
                            input_text=f"Reference document: {source}",
                            output=section.strip(),
                        )
                    )

        return qa_pairs

    def _generate_questions_from_content(self, content: str) -> List[str]:
        """Generate questions from content."""
        # Simplified question generation
        questions = []

        # Check for key phrases to generate relevant questions
        content_lower = content.lower()

        if "battery" in content_lower or "cell" in content_lower:
            questions.append("What do you know about battery cells?")

        if "manufacturing" in content_lower or "process" in content_lower:
            questions.append("Explain the manufacturing process.")

        if "quality" in content_lower or "test" in content_lower:
            questions.append("What quality control procedures are used?")

        if "safety" in content_lower or "safe" in content_lower:
            questions.append("What safety procedures should be followed?")

        if "specification" in content_lower or "parameter" in content_lower:
            questions.append("What are the key specifications?")

        return questions[:2]  # Return max 2 questions per section

    def save_to_jsonl(self, output_path: str):
        """Save examples to JSONL file."""
        with open(output_path, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example) + "\n")

    def load_from_jsonl(self, input_path: str) -> List[Dict]:
        """Load examples from JSONL file."""
        examples = []
        with open(input_path, "r") as f:
            for line in f:
                examples.append(json.loads(line))
        return examples


class BatteryDomainPreparer(DomainDatasetPreparer):
    """Prepare domain-specific dataset for battery manufacturing."""

    def __init__(self):
        super().__init__()
        self._add_seed_examples()

    def _add_seed_examples(self):
        """Add seed Q&A examples for battery manufacturing domain."""

        # Chemistry examples
        self.add_example(
            instruction="Explain the basic chemistry of lithium-ion batteries used in Apple products.",
            input_text="",
            output="Lithium-ion batteries in Apple products use NMC (Nickel Manganese Cobalt) or NCA (Nickel Cobalt Aluminum) cathode chemistry with graphite anodes. The nominal voltage is 3.82V with maximum charge voltage of 4.35V. These batteries offer high energy density (150-250 Wh/kg), low self-discharge, and long cycle life (500+ cycles)."
        )

        self.add_example(
            instruction="What are the main components of a lithium-ion battery?",
            input_text="",
            output="A lithium-ion battery consists of: (1) Cathode (positive electrode) - made of NMC/NCA material, (2) Anode (negative electrode) - graphite/silicon composite, (3) Electrolyte - organic carbonate-based liquid, (4) Separator - polyethylene/polypropylene film, and (5) Current collectors - aluminum (cathode) and copper (anode) foils."
        )

        # Manufacturing examples
        self.add_example(
            instruction="Describe the electrode preparation stage in battery manufacturing.",
            input_text="",
            output="Electrode preparation involves: (1) Mixing - dry mixing cathode active material with conductive carbon and binder (PVDF) at 90:5:5 ratio, (2) Coating - applying mixture to aluminum (cathode) or copper (anode) foil, (3) Calendering - compressing to target density. Environment is controlled at <40% humidity. Coating weight is 20-30 mg/cm² for cathode and 10-15 mg/cm² for anode."
        )

        self.add_example(
            instruction="What is the battery formation process?",
            input_text="",
            output="Formation is the first charge cycle: (1) Charge at 0.1C to 0.5C rate, (2) Voltage limit: 4.2V ± 0.05V, (3) Duration: 12-24 hours, (4) Temperature: 25±2°C. This process forms the solid-electrolyte interphase (SEI) layer. After formation, cells undergo aging for 5-14 days at 25±2°C with voltage checks every 24 hours."
        )

        # Quality control examples
        self.add_example(
            instruction="What tests are performed during quality control?",
            input_text="",
            output="Quality control includes: Electrical tests (capacity at 0.2C discharge ±3%, impedance at 1kHz ±5%, open circuit voltage ±0.02V), Visual inspection (no defects at 10x magnification), Safety tests (external short circuit, overcharge, thermal abuse, penetration, crush). Cells are graded: Grade A (±1% capacity, ±2% impedance), Grade B (±2% capacity, ±4% impedance), Grade C (±3% capacity, ±5% impedance)."
        )

        # Safety examples
        self.add_example(
            instruction="What safety measures are in place for battery handling?",
            input_text="",
            output="Safety measures include: (1) PPE - safety glasses, nitrile gloves, lab coat, face shield, ESD protection, (2) Handling - no puncture/crush/abuse, keep away from heat/flames, use approved chargers, (3) Storage - 20±5°C, <65% RH, ventilation, fire suppression, (4) Emergency - spill kits, eyewash within 10 seconds, fire extinguishers, evacuation procedures."
        )

        self.add_example(
            instruction="How should battery fires be handled?",
            input_text="",
            output="For lithium-ion fires: (1) Alert personnel and press emergency stop, (2) Activate fire suppression, (3) If safe, disconnect power, (4) Use Class D extinguisher for metal fires or water/foam for others, (5) Apply water for cooling, (6) Maintain safe distance -电池 may reignite, (7) Wait for fire department, (8) Ventilate after. Battery fires produce toxic fumes. Never use water on electrolyte fires."
        )


def prepare_battery_dataset(
    documents: Optional[List[Document]] = None,
    output_path: str = "data/finetune/battery_dataset.jsonl",
) -> List[Dict[str, str]]:
    """
    Prepare a complete fine-tuning dataset for battery manufacturing.

    Args:
        documents: Optional documents to extract QA pairs from
        output_path: Path to save the dataset

    Returns:
        List of training examples
    """
    preparer = BatteryDomainPreparer()

    # Add document-based Q&A if provided
    if documents:
        doc_pairs = preparer.create_qa_pairs_from_documents(documents)
        for pair in doc_pairs:
            preparer.add_example(
                instruction=pair["instruction"],
                input_text=pair["input"],
                output=pair["output"],
            )

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preparer.save_to_jsonl(output_path)

    return preparer.examples
