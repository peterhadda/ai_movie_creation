import os
from pathlib import Path
from typing import List, Dict

import pyttsx3
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
)


# -----------------------------
# CONFIG
# -----------------------------
OUT_DIR = Path("out")
ASSETS_DIR = OUT_DIR / "assets"
AUDIO_DIR = OUT_DIR / "audio"
VIDEO_PATH = OUT_DIR / "mini_movie_30s.mp4"

W, H = 1280, 720
FPS = 24

# A tiny "screenplay" structure:
# Each line becomes a short audio clip, displayed as subtitle on a scene card.
SCRIPT: List[Dict] = [
    {
        "scene": "INT. ROOM – NIGHT",
        "bg": (20, 20, 30),
        "lines": [
            {"speaker": "NARRATOR", "text": "Montreal, 2 A M. The city is quiet… but not for long."},
            {"speaker": "SAM", "text": "Did you hear that?"},
        ],
    },
    {
        "scene": "EXT. STREET – NIGHT",
        "bg": (10, 10, 10),
        "lines": [
            {"speaker": "LINA", "text": "Yeah. Someone’s running."},
            {"speaker": "SAM", "text": "We shouldn’t follow…"},
            {"speaker": "LINA", "text": "We already are."},
        ],
    },
    {
        "scene": "CUT TO BLACK",
        "bg": (0, 0, 0),
        "lines": [
            {"speaker": "NARRATOR", "text": "Next time… we don’t look back."},
        ],
    },
]

# Target total duration (approx). We’ll pad with silence if needed.
TARGET_SECONDS = 30.0


# -----------------------------
# TTS (offline) using pyttsx3
# -----------------------------
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)  # speech speed
    engine.setProperty("volume", 1.0)
    return engine


def tts_to_wav(engine, text: str, out_path: Path):
    """
    Saves TTS audio to a .wav file (offline).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Some OS voices behave differently; this is the most compatible approach:
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()


# -----------------------------
# SIMPLE SCENE IMAGE GENERATOR
# -----------------------------
def make_scene_image(scene_title: str, subtitle: str, bg_rgb, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (W, H), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Try a default font; if not found, Pillow uses a fallback.
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 48)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Title at top
    draw.text((60, 50), scene_title, font=font_title, fill=(230, 230, 230))

    # Subtitle box at bottom
    box_h = 160
    box_y0 = H - box_h - 40
    draw.rectangle([40, box_y0, W - 40, H - 40], fill=(0, 0, 0))

    # Subtitle text
    wrapped = wrap_text(subtitle, font_sub, max_width=W - 140, draw=draw)
    draw.text((70, box_y0 + 35), wrapped, font=font_sub, fill=(255, 255, 255))

    img.save(out_path)


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current = []
    for w in words:
        test = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# -----------------------------
# BUILD MOVIE
# -----------------------------
def build_movie():
    OUT_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    engine = init_tts()

    video_clips = []
    audio_clips = []

    clip_index = 0

    for scene in SCRIPT:
        scene_title = scene["scene"]
        bg = scene["bg"]

        for line in scene["lines"]:
            speaker = line["speaker"]
            text = line["text"]

            # 1) Create audio
            wav_path = AUDIO_DIR / f"line_{clip_index:03d}.wav"
            tts_to_wav(engine, f"{speaker}. {text}", wav_path)

            aclip = AudioFileClip(str(wav_path))
            audio_clips.append(aclip)

            # 2) Create matching image (subtitle card)
            img_path = ASSETS_DIR / f"frame_{clip_index:03d}.png"
            subtitle = f"{speaker}: {text}"
            make_scene_image(scene_title, subtitle, bg, img_path)

            # 3) Make a video clip with the same duration as audio
            vclip = ImageClip(str(img_path)).set_duration(aclip.duration)
            video_clips.append(vclip)

            clip_index += 1

    # Combine everything
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_audio = concatenate_audioclips(audio_clips)
    final_video = final_video.set_audio(final_audio)

    # Pad or trim to 30 seconds
    dur = final_video.duration
    if dur < TARGET_SECONDS:
        # Add a final black screen with silence
        pad = TARGET_SECONDS - dur
        black = Image.new("RGB", (W, H), (0, 0, 0))
        pad_img = ASSETS_DIR / "pad_black.png"
        black.save(pad_img)

        pad_clip = ImageClip(str(pad_img)).set_duration(pad)
        final_video = concatenate_videoclips([final_video, pad_clip], method="compose")
    else:
        final_video = final_video.subclip(0, TARGET_SECONDS)

    # Export
    final_video.write_videofile(
        str(VIDEO_PATH),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
    )

    print(f"✅ Done! Saved to: {VIDEO_PATH}")


if __name__ == "__main__":
    build_movie()
