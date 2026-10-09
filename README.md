# Driver Drowsiness Detection

A real-time drowsiness detection system I built using just a laptop webcam. It watches your eyes and mouth, sounds an alarm if your eyes stay closed for too long, and sends a photo alert straight to your phone on Telegram.

I'm an ECE student, and I wanted to build something that combines signal processing ideas with computer vision. Drowsy driving is a real cause of road accidents, so this felt like a problem worth trying to solve, even at a small scale.

## Screenshots

| Awake | Drowsy alert | Yawning |
|-------|--------------|---------|
| ![Awake](screenshots/awake.png) | ![Drowsy](screenshots/drowsy.png) | ![Yawn](screenshots/yawn.png) |

### Phone alert on Telegram

![Telegram Alert](screenshots/telegram_alert.png)

## What it does

- Tracks your eyes and mouth in real time using the webcam
- Tells the difference between a normal blink and your eyes actually closing
- Sounds an alarm and turns the screen red when you look drowsy
- Detects yawns and counts blinks
- Sends a photo of you with the time to your phone through a Telegram bot
- Logs every drowsy event and yawn to a CSV file so you can look at it later

## How it works

**Face tracking:** MediaPipe Face Mesh finds 468 points on the face in every frame. I only use 6 points around each eye and 4 around the mouth.

**Eye Aspect Ratio (EAR):** this compares how tall the eye is to how wide it is.

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 × |p1 - p4|)
```

When my eyes are open, EAR is around 0.33. When they close, it drops close to 0. If it stays low for more than 1.5 seconds, that's not a blink anymore, so the alarm goes off.

**Mouth Aspect Ratio (MAR):** mouth height divided by mouth width. It jumps up when you yawn, and if it stays above 0.6 for a second, it counts as a yawn.

**Calibration:** one thing I noticed is that everyone's eyes are shaped differently, so a fixed threshold doesn't work well for everyone. For the first 3 seconds, the program measures your normal open-eye EAR and sets the threshold at 75% of it. For me, that came out to about 0.26.

**Phone alerts:** when drowsiness is detected, a snapshot is sent to my phone through the Telegram Bot API. At first, sending the photo froze the video for a moment, so I moved it to a background thread. There's also a 30-second cooldown so it doesn't spam the phone.

## Built with

- Python 3.10
- OpenCV for the webcam and display
- MediaPipe Face Mesh for facial landmarks
- NumPy for the distance calculations
- Pygame for the alarm sound (the beep is generated in code, so no sound file is needed)
- Requests + Telegram Bot API for phone alerts

## Running it yourself

```bash
git clone https://github.com/laasya135/drowsiness-detection.git
cd drowsiness-detection
python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on Mac/Linux
pip install -r requirements.txt
python drowsiness_detection.py
```

Keep your eyes open for the first 3 seconds while it calibrates, then it starts detecting. Press **q** to quit.

**For phone alerts:** make a bot with @BotFather on Telegram, get your chat ID, and paste both near the top of `drowsiness_detection.py`. If you skip this, everything else still works; you just won't get phone alerts.

## What I learned

- How facial landmark detection works, and how a simple ratio like EAR can be surprisingly reliable
- Why thresholds need calibration instead of hardcoded values
- Using threads so slow network calls don't block real-time video
- Working with an external API and keeping secrets like bot tokens out of public code

## Limitations

- Doesn't work well in low light or with a bright window behind you
- Sunglasses or thick-framed glasses can block the eye landmarks
- Turning your head too far to the side makes tracking unreliable
- Phone alerts need an internet connection (the local alarm works offline)

## What I'd like to add next

- Run it on a Raspberry Pi with an IR camera so it works at night in a real car
- Detect head nodding as another sign of drowsiness
- Send alerts to an emergency contact too, not just the driver

## About me

**Laasya G**, ECE student at JSS Academy of Technical Education, Bengaluru
[LinkedIn](https://linkedin.com/in/laasya-g-28642832a) · [GitHub](https://github.com/laasya135)
