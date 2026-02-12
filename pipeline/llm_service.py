"""LLM service using Google Gemini API."""

from collections.abc import Iterator

import google.generativeai as genai


class LLMService:
    """Service for generating responses using Google Gemini."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-2.5-flash")

    def _build_prompt(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        """Build a strongly structured, context-aware prompt."""
        history_block = ""
        if chat_history:
            history_block = "\n\nPrevious conversation:\n"
            for msg in chat_history[-10:]:  # Last 10 turns to avoid overflow
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_block += f"{role}: {msg.get('content', '')}\n"

        return f"""You are a precise assistant that answers questions using the provided document context and conversation history when relevant.

Context from documents:
{context}
{history_block}

Current question: {question}

INSTRUCTIONS:

Step 1 — Determine intent:
- If the question is conversational (greetings, thanks, casual chat), respond naturally in 1–2 lines and DO NOT reference the documents.
- If the question requires knowledge from the uploaded documents, answer strictly using the retrieved document context.

Step 2 — Output Style and Formatting (STRICT):
- Write in a formal, academic, report-style tone.
- Do NOT use Markdown headings (###, ##).
- Do NOT use asterisks (*), hyphens, or bullet symbols.
- Do NOT use compact bullet points.
- CRITICAL: Ensure all output text is clean, human-readable, and properly spaced. Never concatenate words.
- Always include proper whitespace between words and sentences.

- Structure answers as follows:
  1. Begin with 1–2 introductory lines that summarize the topic.
  2. Present key points as clearly separated numbered or roman-numeral sections (i., ii., iii., etc.).
  3. Each point must follow this format exactly:
     - Point title on the same line, followed by a colon.
     - A full explanatory paragraph (3–5 sentences) explaining what it is, how it works, and why it matters.
  4. Leave a blank line between every point and between paragraphs.

- Ensure strong visual spacing between ideas, similar to an academic PDF or project report.
- Avoid overly short explanations and avoid wall-of-text paragraphs.
- Target moderate depth (approximately 12–20 lines total unless the user asks otherwise).
- Avoid meta phrases like “The provided documents mention…”. Write directly and authoritatively.

SPECIAL PATTERN — questions about objectives / goals / aims:
- If the question asks about objectives, goals, aims, or core objectives of a project/module/system:
  - First, write a single concise sentence summarizing the overall set of objectives.
  - Then output the objectives as a clean list in this style:
    i. Objective title:
       Paragraph explaining this objective (3–4 sentences, rewritten in your own words).

    ii. Next objective title:
        Paragraph explaining this objective.

    iii. Next objective title:
         Paragraph explaining this objective.
- Include only actual objectives from the documents; omit unrelated background or methodology unless needed
  to clarify an objective.

Step 3 — Grounding:
- Use the document context as the primary source.
- If some information is missing, explicitly state that it is not covered in the documents.
- If external knowledge is used, clearly mark it as (model knowledge).

Additional rules:
- Do NOT repeat the entire document context.
- Rewrite any dense or formatted source content into clean, paragraph-based academic prose before answering.

Citation rules (STRICT):
- Before explaining a point, first include an exact quote from the context, in double quotes.
- On the very next line, and on that line ONLY, show the citation in this format:
  Source: [<document_name>](#), Page <page_number>
- Leave a blank line after the Source line, then continue your explanation in a new paragraph.
- Never place any explanatory text on the same line as the Source.
- You may use multiple quotes from the same document and page when it helps the explanation.
- Never mention or invent chunk IDs, vector scores, or any internal identifiers in the answer.

When information is missing:
- If the documents do not contain the requested information, explicitly answer:
  "Not found in the document."
- You may optionally add general background as (model knowledge), clearly separated from the cited content.
"""


    def _extract_text_from_chunk(self, chunk) -> str:
        """Extract text from a response chunk (handles both simple and multi-part)."""
        try:
            return chunk.text or ""
        except ValueError:
            # Use parts when .text doesn't work for non-simple responses
            text_parts = []
            if chunk.candidates:
                content = chunk.candidates[0].content
                if content and content.parts:
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
            return "".join(text_parts)

    def generate(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        """Generate an answer given a question, retrieved context, and optional chat history."""
        prompt = self._build_prompt(question, context, chat_history)
        response = self._model.generate_content(prompt)
        text = self._extract_text_from_chunk(response)
        return text.strip() if text else "I couldn't generate a response."

    # def generate_stream(
    #     self,
    #     question: str,
    #     context: str,
    #     chat_history: list[dict] | None = None,
    # ) -> Iterator[str]:
    #     """
    #     Stream response text as it is generated.

    #     IMPORTANT: Do not split into words, because that destroys whitespace/newlines
    #     and collapses structured formatting (e.g., roman numeral sections).
    #     """
    #     prompt = self._build_prompt(question, context, chat_history)
    #     response = self._model.generate_content(prompt, stream=True)
    #     for chunk in response:
    #         text = self._extract_text_from_chunk(chunk)
    #         if text:
    #             # Yield raw text to preserve formatting/newlines.
    #             yield text


    def generate_stream(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> Iterator[str]:

        prompt = self._build_prompt(question, context, chat_history)

        try:
            print("Calling LLM now...")

            response = self._model.generate_content(prompt, stream=True)

            for chunk in response:
                try:
                    text = self._extract_text_from_chunk(chunk)
                    if text:
                        yield text
                except Exception:
                    continue

        except Exception:
            return

