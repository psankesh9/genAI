#!/usr/bin/env python
# coding: utf-8

"""
Combined Prompt Engineering Demo Script

This script demonstrates:
1) Summarizing (based on l4-summarizing.py)
2) Inferring (based on l5-inferring.py)
3) Transforming (based on l6-transforming.py)
4) Expanding (based on l7-expanding.py)
5) Chat-like interactions (based on l8-chatbot.py)
6) An additional Chain-of-Thought example
7) A short demonstration of how to integrate a LoRA fine-tuning flow

Make sure you have installed (at least):
    pip install openai python-dotenv
and that your OpenAI API key is set (e.g., via .env or environment variable).
"""

import os
import openai
from dotenv import load_dotenv

###############################################################################
# 0. Environment and Setup
###############################################################################
_ = load_dotenv()  # Load .env if present (containing OPENAI_API_KEY)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is accessible


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    """
    Generic helper to send a single-turn prompt to the OpenAI API.
    Returns only the 'content' of the model's response.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    """
    Helper for multi-turn (chat-like) conversations,
    where messages is a list of dictionaries with 'role' and 'content'.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


###############################################################################
# 1. Summarizing
###############################################################################
def demo_summarizing():
    """
    Demonstrates summarization with a short product review text.
    We ask for a summary focusing on a particular dimension (price/value).
    """
    review_text = (
        "Got this panda plush toy for my daughter's birthday. "
        "She loves it and takes it everywhere. It's soft, super cute, "
        "and the face is very friendly. It's a bit small for what I paid, "
        "and there might be other options that are bigger for the same price."
    )

    prompt = f"""
    Your task is to generate a short summary of a product review from an ecommerce site
    focused on price and perceived value. Summarize the review below in at most 30 words.
    Review: ```{review_text}```
    """
    summary = get_completion(prompt)
    print("=== Summarizing Demo ===")
    print(f"Review:\n{review_text}\n")
    print(f"Summary:\n{summary}\n")
    print("========================================\n")


###############################################################################
# 2. Inferring
###############################################################################
def demo_inferring():
    """
    Demonstrates inference about sentiment (positive/negative)
    and extraction of product/company from a short review.
    """
    lamp_review = (
        "Needed a nice lamp for my bedroom. Got it fast. The string to the lamp broke "
        "during shipping and the company happily sent a new one. It was easy to assemble. "
        "Missing part was replaced quickly. Great company!"
    )

    prompt_sentiment = f"""
    What is the sentiment of the following product review?
    Review: ```{lamp_review}```
    Give your answer as a single word, either "positive" or "negative".
    """
    sentiment = get_completion(prompt_sentiment)

    prompt_extraction = f"""
    Identify the following items from the review text: 
    - Item purchased by reviewer
    - Company that made the item

    The review is delimited with triple backticks.
    Format your response as JSON with "Item" and "Brand" as keys.
    Review text: ```{lamp_review}```
    """
    extraction = get_completion(prompt_extraction)

    print("=== Inferring Demo ===")
    print(f"Review:\n{lamp_review}\n")
    print(f"Sentiment:\n{sentiment}")
    print(f"Extraction:\n{extraction}\n")
    print("========================================\n")


###############################################################################
# 3. Transforming
###############################################################################
def demo_transforming():
    """
    Demonstrates transformations:
    - Converting text to another style/tone
    - Converting data formats (JSON to HTML)
    """
    # Example: Tone transformation from slang to a formal letter
    slang_text = "Dude, this is Joe, check out this spec on the standing lamp."
    prompt_tone = f"""
    Translate the following from slang to a formal business letter:
    '{slang_text}'
    """
    formal_text = get_completion(prompt_tone)

    # Example: JSON to HTML table
    data_json = {
        "restaurant_employees": [
            {"name": "Shyam", "email": "shyam@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
            {"name": "Jai", "email": "jai@example.com"},
        ]
    }
    prompt_html = f"""
    Translate the following python dictionary from JSON to an HTML table
    with column headers and title: {data_json}
    """
    html_table = get_completion(prompt_html)

    print("=== Transforming Demo ===")
    print(f"Original (slang): {slang_text}")
    print(f"Transformed (formal): {formal_text}\n")
    print("JSON to HTML Table:\n", html_table)
    print("========================================\n")


###############################################################################
# 4. Expanding
###############################################################################
def demo_expanding():
    """
    Demonstrates an example of using the model to 'expand' or
    produce longer text, e.g., generating a custom support reply.
    """
    review_text = (
        "I bought the 17-piece blender system on seasonal sale for $49, "
        "but a few weeks later the price went up to about $70-89. It feels like price gouging. "
        "After a year, the motor started making noise and the warranty was already expired."
    )

    # If sentiment was negative, we might want to apologize and offer support
    sentiment = "negative"  # Hardcoded for demo
    prompt_reply = f"""
    You are a customer service AI assistant.
    The customer review is: ```{review_text}```
    Sentiment: {sentiment}

    Write a short, professional email reply. If sentiment is negative, apologize and
    suggest they can reach out to customer service. Thank them for the details they provided.
    Sign the email as 'AI Customer Agent'.
    """
    email_reply = get_completion(prompt_reply, temperature=0.7)

    print("=== Expanding Demo ===")
    print(f"Customer Review:\n{review_text}\n")
    print(f"Draft Email Reply:\n{email_reply}\n")
    print("========================================\n")


###############################################################################
# 5. Chat-like Interaction & Chain-of-Thought
###############################################################################
def demo_chat_and_cot():
    """
    Demonstrates a mini chat flow and a chain-of-thought example
    where the model is asked to reason step-by-step for a math problem.
    """
    # A short chat example
    messages = [
        {"role": "system", "content": "You are a friendly chatbot."},
        {"role": "user", "content": "Hello, I need help picking a laptop."},
        {"role": "assistant", "content": "Sure! What is your budget and primary use?"},
        {
            "role": "user",
            "content": (
                "I have about $700 and I need it mostly for web browsing and "
                "light office work. Maybe some streaming."
            ),
        },
    ]
    response_chat = get_completion_from_messages(messages, temperature=1.0)

    # A chain-of-thought (CoT) prompt example
    cot_prompt = """
    You are a brilliant mathematician. Please reason step-by-step to answer accurately.
    Question: If a plane travels 500 km at 250 km/h, how many hours does it take?
    Let's break down the reasoning first, then provide the final answer:
    """
    cot_response = get_completion(cot_prompt, temperature=0.3)

    print("=== Chat & Chain-of-Thought Demo ===")
    print("--- Chat-like Interaction ---")
    print("User final question: 'I have about $700 and need it for web/office/streaming'")
    print(f"Assistant response:\n{response_chat}\n")

    print("--- Chain-of-Thought (CoT) for a math question ---")
    print(cot_response)
    print("========================================\n")


###############################################################################
# MAIN DEMO RUNNER
###############################################################################
def main():
    demo_summarizing()
    demo_inferring()
    demo_transforming()
    demo_expanding()
    demo_chat_and_cot()
    demo_lora_code_snippet()


if __name__ == "__main__":
    main()
