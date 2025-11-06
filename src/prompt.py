
from langchain_core.prompts import ChatPromptTemplate


# Main prompt template for medical questions
MEDICAL_PROMPT_TEMPLATE = """You are a Medical expert assistant. Provide accurate and concise information based on the provided documents.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer using the context above. If you don't know the answer based on the context, say so. Keep your answer to 3 sentences maximum."""


def get_medical_prompt() -> ChatPromptTemplate:
    """
    Get the ChatPromptTemplate for medical questions.
    
    Returns:
        ChatPromptTemplate configured with the medical prompt
    """
    return ChatPromptTemplate.from_template(MEDICAL_PROMPT_TEMPLATE)


# Alternative prompt templates for different use cases
DETAILED_PROMPT_TEMPLATE = """You are a Medical expert assistant. Provide detailed and accurate information based on the provided documents.

Context:
{context}

Question: {question}

Answer: Provide a comprehensive answer using the context above. Include relevant details, symptoms, treatments, or other pertinent information. If you don't know the answer based on the context, clearly state that."""


def get_detailed_prompt() -> ChatPromptTemplate:
    """
    Get a more detailed ChatPromptTemplate for complex medical questions.
    
    Returns:
        ChatPromptTemplate configured with the detailed prompt
    """
    return ChatPromptTemplate.from_template(DETAILED_PROMPT_TEMPLATE)


SIMPLE_PROMPT_TEMPLATE = """Based on the following context, answer the question briefly.

Context: {context}

Question: {question}

Answer:"""


def get_simple_prompt() -> ChatPromptTemplate:
    """
    Get a simple ChatPromptTemplate for quick answers.
    
    Returns:
        ChatPromptTemplate configured with the simple prompt
    """
    return ChatPromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE)
