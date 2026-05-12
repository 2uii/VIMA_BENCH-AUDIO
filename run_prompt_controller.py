import json

# load prompt from UI
with open("prompt.json", "r") as f:
    prompt = json.load(f)

task = prompt["task"]
obj = prompt["object"]
color = prompt["color"]
audio = prompt["audio_identity"]

print("=== Audio-VIMA Prompt Controller ===")
print("Task:", task)
print("Target object:", obj)
print("Color:", color)
print("Audio identity:", audio)

# convert UI prompt into a simple internal experiment description
experiment = {
    "task_selection": task,
    "target_object": obj,
    "target_color": color,
    "target_audio": audio,
    "query_text": f"Select the {color} {obj} with audio identity {audio}",
}

print("\nGenerated experiment config:")
print(experiment)
