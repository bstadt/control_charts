"""Agent implementation with RAG database and LLM backend."""

from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI

from .config import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from .database import VectorDatabase, QAPair


@dataclass
class Agent:
    """An agent with a RAG database and LLM backend."""

    id: int
    database: VectorDatabase
    model: str = "gpt-4o-mini"
    retrieval_k: int = 5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    # Question tracking
    questions_in_play: set[str] = field(default_factory=set)
    known_questions: set[str] = field(default_factory=set)

    # OpenAI client (initialized lazily)
    _client: OpenAI | None = field(default=None, repr=False)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def unknown_questions(self) -> set[str]:
        """Questions in play that this agent doesn't know."""
        return self.questions_in_play - self.known_questions

    def initialize_knowledge(self, qa_pairs: list[QAPair]) -> None:
        """Initialize agent with a set of QA pairs, marking them as known."""
        self.database.add_many(qa_pairs)
        for qa in qa_pairs:
            self.known_questions.add(qa.question)

    def set_questions_in_play(self, questions: set[str]) -> None:
        """Set the questions that are in play for this simulation."""
        self.questions_in_play = questions

    def mark_known(self, question: str) -> None:
        """Mark a question as known."""
        self.known_questions.add(question)

    def select_question_to_ask(self, rng: np.random.Generator) -> str | None:
        """Randomly select a question from unknown set to ask a peer."""
        unknown = list(self.unknown_questions)
        if not unknown:
            return None
        return rng.choice(unknown)

    def answer(self, question: str, question_embedding: np.ndarray) -> str:
        """Answer a question using RAG retrieval and LLM."""
        # Retrieve relevant QA pairs
        retrieved = self.database.search(question_embedding, k=self.retrieval_k)

        # Format context
        if retrieved:
            context_parts = []
            for qa in retrieved:
                context_parts.append(f"Q: {qa.question}\nA: {qa.answer}")
            context = "\n\n".join(context_parts)
        else:
            context = "(No relevant information found)"

        # Build prompt
        user_prompt = self.prompt_template.format(
            retrieved_context=context,
            question=question
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def receive_answer(self, question: str, answer: str, question_embedding: np.ndarray) -> bool:
        """Receive an answer from a peer. Returns True if knowledge was added."""
        # Check if this is a valid answer (not "I don't know")
        if self._is_dont_know(answer):
            return False

        # Always insert (no duplicate checking per spec)
        qa_pair = QAPair(
            question=question,
            answer=answer,
            embedding=question_embedding
        )
        self.database.add(qa_pair)
        self.mark_known(question)
        return True

    def _is_dont_know(self, answer: str) -> bool:
        """Check if an answer is a 'don't know' response."""
        lower = answer.lower().strip()
        dont_know_phrases = [
            "i don't know",
            "i do not know",
            "i dont know",
            "i don't have",
            "i do not have",
            "no relevant",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not enough information",
        ]
        return any(phrase in lower for phrase in dont_know_phrases)

    def get_state(self) -> dict:
        """Get agent state for logging/debugging."""
        return {
            "id": self.id,
            "db_size": len(self.database),
            "known_count": len(self.known_questions),
            "unknown_count": len(self.unknown_questions),
            "in_play_count": len(self.questions_in_play),
        }
