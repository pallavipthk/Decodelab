"""
DecodeLabs - Week 1 - Project 1: Rule-Based AI Chatbot (Enhanced)
Fully self-contained. No external files required.
Run: python Week1.py
"""

import random
from datetime import datetime

responses = {
    "hello": ["Hi there! How can I help you today?", "Hey! Good to see you."],
    "hi": ["Hello! Welcome to DecodeLabs support bot.", "Hi! What's up?"],
    "how are you": ["I'm just a program, but I'm running smoothly! How about you?",
                     "Doing great, thanks for asking!"],
    "what is your name": ["I'm DecodeBot, your friendly rule-based assistant."],
    "what can you do": ["I can chat using predefined rules, remember your name, "
                         "and greet you based on the time of day."],
    "help": ["Try greeting me, asking my name, or how I'm doing. Type 'bye' to exit."],
    "bye": ["Goodbye! Have a great day!", "See you later!"],
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["No problem!", "Glad to help!"],
    "who made you": ["I was built by an intern at DecodeLabs as part of Project 1!"],
    "what is ai": ["AI is the field of making machines simulate intelligent behavior. "
                   "I'm the 'rule-based' flavor of it!"],
    "what is python": ["Python is the programming language I'm written in — "
                       "great for beginners and AI alike."],
    "joke": ["Why do programmers prefer dark mode? Because light attracts bugs!"],
    "weather": ["I can't check live weather yet, but I hope it's sunny where you are!"],
    "sorry": ["No worries at all!"],
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}
FALLBACK_RESPONSE = "I do not understand. Type 'help' to see what I can do."

KEYWORD_RULES = [
    ("name", "what is your name"),
    ("joke", "joke"),
    ("weather", "weather"),
    ("python", "what is python"),
    ("ai", "what is ai"),
]

session_state = {"user_name": None}


def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()


def time_based_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! ☀️ How can I help you today?"
    elif hour < 18:
        return "Good afternoon! How can I help you today?"
    else:
        return "Good evening! How can I help you today?"


def try_remember_name(clean_input: str):
    if clean_input.startswith("my name is "):
        name = clean_input.replace("my name is ", "").strip().title()
        session_state["user_name"] = name
        return f"Nice to meet you, {name}! I'll remember that."
    return None


def get_response(clean_input: str) -> str:
    name_reply = try_remember_name(clean_input)
    if name_reply:
        return name_reply

    if clean_input in ("hello", "hi") and session_state["user_name"]:
        base = random.choice(responses[clean_input])
        return f"{base} Welcome back, {session_state['user_name']}!"

    if clean_input in ("hello", "hi"):
        return time_based_greeting()

    if clean_input in responses:
        return random.choice(responses[clean_input])

    for keyword, mapped_key in KEYWORD_RULES:
        if keyword in clean_input:
            return random.choice(responses[mapped_key])

    return FALLBACK_RESPONSE


def run_chatbot():
    print("=" * 55)
    print(" DecodeBot v2 - Rule-Based AI Chatbot (Enhanced)")
    print(" Type 'exit', 'quit', or 'bye' to end the chat.")
    print(" Tip: try 'my name is <yourname>', 'joke', or 'what is ai'")
    print("=" * 55)

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        if clean_input in EXIT_COMMANDS:
            name = session_state["user_name"]
            farewell = random.choice(responses.get(clean_input, ["Goodbye!"]))
            if name:
                farewell += f" See you next time, {name}!"
            print(f"Bot: {farewell}")
            break

        reply = get_response(clean_input)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_chatbot()
