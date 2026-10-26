# Desk shot: props, shot plan, voice script

Everything on these props is fictional. Phone numbers use the 555-0100–0199 range reserved for fiction, emails use `.example`, and "ApexCorp", "Relay" and "Northwind Bank" are made up. The only real face should be yours, on the badge.

## Props

| Prop | Where | Setup |
|---|---|---|
| `mock_screen.html` | Laptop | Open in the browser, F11 (full screen), zoom 100%. Brightness ~60%. |
| `phone_alert.html` | Phone, face up, right of the laptop | Open on the phone (send yourself the file, or serve this folder: `python -m http.server` and open `http://<laptop-ip>:8000/phone_alert.html`). Set screen timeout to 5 min, brightness ~70%. Tap once just before you start: the first alert slides in 2 s later, the second at 5.5 s. |
| `id_badge.html` | Printed, or on a second screen, left of the keyboard | Click the photo area and pick a photo of yourself (stays in the browser). Print at actual size, or show it scaled up on a tablet. Small in frame on purpose: that's the face a first pass can miss. |

## Camera

- Landscape, 1080p, **30 fps**, HDR off. Not 4K (the pipeline scales to 1280 px anyway).
- Tripod or elbows on the desk; 1–2 ft away, about 15° off straight-on (cuts the moiré lines screens make).
- Long-press on the laptop text to lock focus and exposure before you hit record.
- Quiet room; the phone's mic is fine at this distance.

## Shot plan (15 s)

| Time | Picture | You say |
|---|---|---|
| 0–3 s | Whole desk: laptop memo readable, phone dark-ish, badge visible | (silence) |
| 1 s | Tap the phone off-camera (alert arrives at ~3 s) | |
| 3–12 s | Slow drift from the laptop table → the phone alert → the badge | The script below |
| 12–15 s | Settle back on the whole desk, hold still | (silence) |

## Voice script (~10 s, natural pace)

> "Okay, before the review I need to call **Elena Rostova** at **four-one-five, five-five-five, zero-one-four-two**. Her offer's **two hundred fourteen thousand**. And loop in **David Miller**, employee ID **nine-oh-two-one-four**."

Every bold item is also on screen, so the audio redaction (T10) has something to bleep that matches what's blacked out on the laptop, phone and badge.

## Instructions to run on this clip

- Suggested: `Black out the laptop screen and the phone screen, and blur every face.`
- Follow-up: `Also blur the ID badge.`

Add it with:

```
cd backend
.venv/Scripts/python -m app.cache add-clip <recording> --name desk_mine --title "My desk" ^
  --prompt "Black out the laptop screen and the phone screen, and blur every face." ^
  --follow-up "Also blur the ID badge."
```

(or upload it in the Colab notebook's "Your own clip" cell with the same instructions).
