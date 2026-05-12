import json
import pickle

# load UI prompt
with open("prompt.json", "r") as f:
    prompt = json.load(f)

task = prompt["task"]
target_object = prompt["object"]
target_color = prompt["color"]
target_audio = prompt["audio_identity"].replace(".wav", "")

# load training samples
samples = pickle.load(open("audio_training_samples.pkl", "rb"))

matches = []

for s in samples:
    audio_id = s["audio_id"]
    parts = audio_id.split("__", 1)

    obj_name = parts[0] if len(parts) > 0 else ""
    obj_color = parts[1] if len(parts) > 1 else ""
    wav_path = s.get("wav_path", "")

    # match by object, color substring, and wav/audio identity
    if (
        target_object in obj_name
        and target_color in obj_color
        and target_audio in wav_path
    ):
        matches.append({
            "episode": s["episode"],
            "placeholder": s["placeholder"],
            "audio_id": s["audio_id"],
            "wav_path": wav_path,
        })

print("=== Prompt Matcher ===")
print("Task:", task)
print("Target object:", target_object)
print("Target color:", target_color)
print("Target audio:", target_audio)
print("\nMatches found:", len(matches))

for m in matches[:10]:
    print(m)
