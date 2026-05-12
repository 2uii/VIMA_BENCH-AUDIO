import json
import pickle

# load prompt
with open("prompt.json", "r") as f:
    prompt = json.load(f)

target_object = prompt["object"]
target_color = prompt["color"]
target_audio = prompt["audio_identity"].replace(".wav", "")

samples = pickle.load(open("audio_training_samples.pkl", "rb"))

matches = []

for s in samples:
    audio_id = s["audio_id"]
    parts = audio_id.split("__", 1)
    obj_name = parts[0] if len(parts) > 0 else ""
    obj_color = parts[1] if len(parts) > 1 else ""
    wav_path = s.get("wav_path", "")

    score = 0

    if target_object in obj_name:
        score += 1
    if target_color in obj_color:
        score += 1
    if target_audio in wav_path:
        score += 1

    if score > 0:
        matches.append({
            "episode": s["episode"],
            "placeholder": s["placeholder"],
            "audio_id": s["audio_id"],
            "wav_path": wav_path,
            "score": score,
        })

matches.sort(key=lambda x: x["score"], reverse=True)

print("=== Prompt Selector ===")
print("Prompt:", prompt)
print("\nTop candidates:")

for m in matches[:10]:
    print(m)

if matches:
    best = matches[0]
    print("\nSelected candidate:")
    print(best)
else:
    print("\nNo candidate selected.")
