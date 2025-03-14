# human_in_loop/ai_human_feedback.py
import logging
import json

logging.basicConfig(filename='logs/ai_human_feedback.log', level=logging.INFO)

feedback_data = []

def collect_human_feedback(user_input):
    feedback_data.append(user_input)
    with open("logs/human_feedback.json", "w") as f:
        json.dump(feedback_data, f, indent=4)
    logging.info(f"Collected human feedback: {user_input}")
    return "Feedback recorded."

print("AI Human-in-the-Loop Feedback System Initialized.")