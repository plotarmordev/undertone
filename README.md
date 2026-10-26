# Undertone

[![doc: Apache-2.0](https://img.shields.io/badge/doc-Apache--2.0-0969da?style=flat-square)](#license-and-notice)
[![weights: FACL v1.0](https://img.shields.io/badge/weights-FACL_v1.0-8250df?style=flat-square)](#license-and-notice)
[![works with: drama-tensorfold](https://img.shields.io/badge/works_with-drama--tensorfold-76b900?style=flat-square)](#applying-it-to-drama-tensorfold)
[![status: evidence from my renders](https://img.shields.io/badge/status-evidence_from_my_renders-a16207?style=flat-square)](#honest-caveats)

**I tune a cloned Drama 3 voice so a spoken reply sounds like someone talking rather than someone reading.**

This is how I do it: delivery cues that are actually performed, a real pause where the subject changes, breaths inside long
stretches, an even register, a pace that does not rush, and guards that keep a broken take from ever being heard. Every
setting in this README is the value I run, with the reason for it, the evidence I have, and how you can check it on your
own voice. Most of it needs no model at all: plain text rules before synthesis, server flags during it, and a little
audio surgery after it. Nothing here trains or fine-tunes anything.

## 🚀 Quick start

**1. Build the engine and download the weights.** Follow the [drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold)
README's Quickstart: it builds the engine and starts a local server with `POST /v1/audio/speech`. Every server flag on
this page exists, with the same name and meaning, in the engine that `scripts/build-engine.sh --weights nvfp4` builds.
I run the NVFP4 pruned weights, [plotarmordev/drama-3-NVFP4-Pruned](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned),
downloaded myself from Hugging Face under the Fish Audio Community License, Version 1.0
([License and NOTICE](#license-and-notice)); the full model, [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3),
works too, with the flag mapping in [On the full-model engine](#on-the-full-model-engine).

**2. Encode a reference clip** of a voice you have the rights to use, with its exact transcript, into a voice file as
the drama-tensorfold README shows under "A reference voice". Put it in the `voices/` directory your launch will pass as
`--voices`: the server reads that directory once, at startup, so the voice files must be there first. How to pick the
clip: [Choosing a reference clip](#choosing-a-reference-clip).

**3. Launch the server** with the flags I run (what each one does: [Every flag, its value, and why](#every-flag-its-value-and-why)):

```sh
python3 recipe/pilot.py --weights nvfp4 --runtime-dir runtime --gpu-lock "$GPU_LOCK" --evidence-dir runs/voice-01 \
  -- --voices "$PWD/voices" --default-voice NAME \
     --max-frames 4096 --long-form --long-form-chars 2400 \
     --cfg-gamma 1.5 --cfg-batched \
     --allowed-controls 'concerned,curious,warmly,sighs,small laugh,laughs' \
     --silence-onset-s 3.0 --silence-mid-s 3 \
     --silence-redraws 2 --silence-redraw-plain --silence-redraw-plain-scope controls \
     --short-reply-ratio 0.5 \
     --register-redraw-st 4 --register-redraw-high 20 --register-high-hz HZ
```

`NAME` is your voice file, and `HZ` the high pitch line for your voice: a few semitones above the median speaking pitch
of your reference clip, calibrated on your own replies. Keep the server on loopback; see the license notes on hosted
services in [License and NOTICE](#license-and-notice).

**4. Write for the ear.** Give the writer of each spoken reply — a person or a language model — the rules in
[Write for the ear](#write-for-the-ear): one idea and one number a sentence, 8 to 20 words, few full stops, no labels,
lists, symbols or sentence-initial fillers, a blank line between topics. If the writer wrote no cues or topic marks, the
automatic rules in [Text side](#text-side-writing-for-the-ear-and-automatic-cues) add them. All of this runs on the text
before the request, with no model.

**5. Send one request per reply:** the whole reply, white space collapsed, with `"response_format": "wav"` and the
`X-Deadline-S` header ([how I compute it](#every-flag-its-value-and-why)). Splitting a reply into several requests
restarts the prosody at every join. Keep the audio only when `X-Finish-Reason` is `stop`; anything else falls back to
text ([Delivery](#delivery-only-a-finished-take-is-sent)).

**6. Shape the audio:** thinking pauses of 1.1 s at topic shifts, 0.85 s breaths, pacing, and one lossy encode at the
end — all of [Audio side](#audio-side-thinking-pauses-breaths-pacing-encoding). The text and audio steps need Python 3
with numpy, and ffmpeg for the final encode; nothing in them needs a GPU or a model.

**7. Measure and listen blind** on paired takes — same texts, same seeds, before and after — before keeping any change
([How to measure your own voice](#how-to-measure-your-own-voice)). For measuring I use a speech recognizer (Whisper
large-v3; small.en is enough for a quick check), a speaker encoder (an ECAPA model trained on VoxCeleb) and a pitch
tracker (pYIN, or the plain autocorrelation tracker in [Register and pitch](#register-and-pitch)).

## 🧭 Contents

| I want to... | Go to |
|---|---|
| Make my cloned voice sound alive in minutes | [Quick start](#-quick-start) |
| Have my AI assistant apply Undertone to my setup for me | [📋 Copy this prompt into your AI](#-copy-this-prompt-into-your-ai) |
| See it applied to Drama TensorFold, or to another engine | [Examples](#-examples) |
| Check a flag value, a rule, a dose of evidence | [📖 Detailed reference](#-detailed-reference) |

- [Quick start](#-quick-start)
- [📋 Copy this prompt into your AI](#-copy-this-prompt-into-your-ai)
- [Examples](#-examples)
  - [Applying it to Drama TensorFold](#applying-it-to-drama-tensorfold)
  - [Applying it to a generic TTS engine](#applying-it-to-a-generic-tts-engine)
- [📖 Detailed reference](#-detailed-reference)
  - [The pipeline at a glance](#the-pipeline-at-a-glance)
  - [Choosing a reference clip](#choosing-a-reference-clip)
  - [Server side: the engine flags](#server-side-the-engine-flags)
  - [Text side: writing for the ear and automatic cues](#text-side-writing-for-the-ear-and-automatic-cues)
  - [Delivery: only a finished take is sent](#delivery-only-a-finished-take-is-sent)
  - [Audio side: thinking pauses, breaths, pacing, encoding](#audio-side-thinking-pauses-breaths-pacing-encoding)
  - [What did not work](#what-did-not-work)
  - [How to measure your own voice](#how-to-measure-your-own-voice)
  - [Reproduction checklist](#reproduction-checklist)
  - [Honest caveats](#honest-caveats)
  - [License and NOTICE](#license-and-notice)

## 📋 Copy this prompt into your AI

Paste this whole block into Claude, ChatGPT, Codex or whichever assistant you use, on the machine that runs your voice
setup, after putting your recipe or engine where it can read it:

```text
I want my text-to-speech voice to sound like someone talking, not someone reading. Apply Undertone,
a voice-tuning recipe, to my TTS setup.

Read these first, in this order:
1. The Undertone README at https://github.com/plotarmordev/undertone — the intro paragraph and the
   quick start, then the sections by anchor: #server-side-the-engine-flags,
   #text-side-writing-for-the-ear-and-automatic-cues, #delivery-only-a-finished-take-is-sent,
   #audio-side-thinking-pauses-breaths-pacing-encoding, #what-did-not-work,
   #how-to-measure-your-own-voice, #honest-caveats and #license-and-notice.
2. The NOTICE file in that repository.
3. My TTS setup: <PATH TO MY RECIPE>. If it is Drama 3 on TensorFold, it is built from
   https://github.com/plotarmordev/drama-tensorfold — read that README's Quickstart, its
   "A reference voice" part and its HTTP contract too.

Then map Undertone onto my setup, tell me what you changed and what my engine cannot support,
and ask me before changing anything outside the voice path. What to map:

- Engine flags: a frame cap with headroom for whole replies (Undertone runs 4096 frames, about
  190 s); one generation per reply, because every join restarts the prosody; guidance toward the
  reference voice (--cfg-gamma 1.5 --cfg-batched); an allowlist of delivery controls (concerned,
  curious, warmly, sighs, small laugh, laughs); silence guards that end a stalled take early
  (--silence-onset-s 3.0, --silence-mid-s 3) and redraw it with the controls removed
  (--silence-redraws 2 --silence-redraw-plain --silence-redraw-plain-scope controls); a
  short-reply trip (--short-reply-ratio 0.5, recalibrated to my voice's characters per second);
  a register-jump redraw (--register-redraw-st 4 --register-redraw-high 20, and
  --register-high-hz a few semitones above my reference voice's median pitch, calibrated on my
  own replies). If my engine is the full-model one, use the flag mapping in the README's server
  section.
- Text side: give my reply writer the "write for the ear" rules (one idea a sentence, 8 to 20
  words, few full stops, no labels, lists, symbols or sentence-initial fillers, blank lines
  between topics); add the six cues automatically only when the writer wrote none, mid-topic,
  about one per 120 characters of speech; mark topic shifts; drop cues that open a topic or sit
  in or right before the last sentence; send each reply as one request with a deadline header
  (20 s + 0.15 s per character, capped at 150 s, sent 3 s less for the audio steps).
- Delivery of finished takes: use the audio only when the take finished cleanly (finish reason
  stop on drama-tensorfold); otherwise let the text reply stand, and log the redraw counts.
- Topic-shift pauses and breaths, on the finished take: lengthen the rest at each topic shift to
  1.1 s, and place a 0.85 s breath after 2 or more sentences and 200 or more characters without
  one, filled with the take's own room tone, never touching speech samples.
- Pacing: cap every other rest at 0.55 s, trim the lead to 0.10 s and the tail to 0.20 s,
  cutting only on the rest grid.
- Encoding: one lossy encode at the end, Opus mono 64 kbit/s VBR audio mode (MP3 96 kbit/s CBR
  only for clients that cannot play Opus); never encode twice.

What NOT to touch: speech samples (only rests get longer or shorter); my reference clip (no
speeding it up, no compression: keep its pauses and its exact transcript); the sampler or seed
settings; the server's redraw logic (never retry from the client: on a deterministic server a
retry returns the identical take); anything that would clone a voice I do not have the rights
to use.

Verify before and after on the same texts and seeds, over many paired takes, and show me the
numbers:
- blind listening first: the same texts in each version, labels hidden, order shuffled, levels
  matched, and I pick;
- word error rate against the text with cues and markers removed (a take over 0.2 has lost
  words; normal takes sit around 1-2 %);
- speaker similarity to my reference clip (a speaker-encoder cosine; a drop of more than 0.02
  is a regression);
- register and pitch (pitch spread, 5-95 % range, phrase velocity, phrase openings against the
  reply's other openings);
- pause statistics (about 1.1 s at topic shifts, 0.85 s at breath points, about 0.55 s at other
  sentence ends, no quiet run of 3 s or more inside speech, time to first speech);
- cue performance (each cued sentence against the same sentence uncued: a longer rest or an
  audible breath, more pitch movement, no hole of 1 s or more at the cue);
- retakes (first-take failures against delivered failures, which should be 0; count redraws in
  any wait figure).
```

## 💡 Examples

### Applying it to Drama TensorFold

This is the path everything in this README was measured on: the NVFP4 pruned Drama 3 weights on one GB10.

| Undertone piece | On drama-tensorfold |
|---|---|
| Engine | Built by its README's Quickstart, `scripts/build-engine.sh --weights nvfp4` |
| Server flags | The launch in [Quick start](#-quick-start), every value explained in [Every flag, its value, and why](#every-flag-its-value-and-why); full-model engines use the mapping in [On the full-model engine](#on-the-full-model-engine) |
| Reference voice | A voice file encoded as that README shows under "A reference voice", chosen per [Choosing a reference clip](#choosing-a-reference-clip) |
| Request | One `POST /v1/audio/speech` per reply, `"response_format": "wav"`, `X-Deadline-S` set |
| Finished take | `X-Finish-Reason: stop`, and nothing else, per [Delivery](#delivery-only-a-finished-take-is-sent) |
| Audio | The steps in [Audio side](#audio-side-thinking-pauses-breaths-pacing-encoding) on the 16-bit mono 44.1 kHz WAV the server returns |
| Check | [How to measure your own voice](#how-to-measure-your-own-voice), then blind listening |

### Applying it to a generic TTS engine

- **Carries over unchanged:** the text rules ([Text side](#text-side-writing-for-the-ear-and-automatic-cues) — plain
  text, no model), the delivery rule (use only takes that finished cleanly, whatever the engine calls that state), the
  audio steps ([Audio side](#audio-side-thinking-pauses-breaths-pacing-encoding) — rests, 1.1 s pauses, 0.85 s breaths,
  pacing, one encode, on whatever WAV the engine returns) and all the measurements.
- **Map to what the engine offers:** the flag table below is my engine's. Look for the same protections: a length cap
  with headroom, one generation per reply, guidance toward the reference voice, a delivery-control allowlist (or strip
  every cue you do not allow), guards that end a stalled take early and redraw it server-side, a short-reply trip
  recalibrated to that engine's own expected characters a second, and a register guard calibrated on that engine's own
  detector and its own log lines, never against outside measurements (the caution in
  [On the full-model engine](#on-the-full-model-engine)).
- **Re-measure before trusting:** which cues stall on the new engine (my cue evidence is from the reference voice on
  Drama 3), where the new voice's normal replies sit for the short-reply ratio, and the high pitch line for the new
  voice.
- **What cannot be moved:** if the engine redraws nothing, a client retry cannot replace it — a deterministic server
  returns the identical take for the same text — so the delivery rule and the measurements are the guards that remain.

## 📖 Detailed reference

### The pipeline at a glance

One spoken reply goes through these steps, in this order. Each one is described below.

1. **Write for the ear** (Text side): the reply's writer follows a few rules about sentences, lists and topics.
2. **Automatic delivery cues** (Text side): if the writer put none in, a word-list rule adds a few, mid-topic.
3. **Topic shifts** (Text side): if the writer marked none, a surface rule marks where a new subject starts.
4. **Cue drops** (Text side): cues that open a topic, or sit in or right before the last sentence, are removed.
5. **One request** (Text side): the whole reply, white space collapsed, goes to the server in a single request with a
   deadline header.
6. **Server** (Server side): keeps only the allowlisted delivery controls, draws with guidance, ends collapsed takes
   early, redraws them without controls, and redraws a take whose register jumps.
7. **Delivery rule** (Delivery): only a take that finished `stop` is used; anything else falls back to text.
8. **Thinking pauses and breaths** (Audio side): the rest the take made at each topic shift becomes 1.1 s, and at a
   breath point 0.85 s, filled with the take's own room tone.
9. **Pacing** (Audio side): every other long rest is shortened, and the lead and tail trimmed.
10. **Encode** (Audio side): one lossy encode at settings that keep the voice's air.

### Choosing a reference clip

The reference clip sets the timbre, and much of the pace and breathiness, of everything the clone says. General advice:

- **Rights first.** Use only a voice you have the right to clone: your own, one recorded with its owner's consent on
  record, or one the model invented. The license forbids imitating an identifiable person without consent.
- **Clean and single-speaker.** No music, no second voice, no strong room echo, one microphone throughout.
- **Natural pace.** Do not speed the clip up to make the clone faster: that gives pace but strips the breathiness
  ([What did not work](#what-did-not-work)). Pace comes more safely from the text ([Write for the ear](#write-for-the-ear)).
- **Whole words.** Start and end the clip at silences, never inside a word.
- **Keep the air.** Do not compress or remove the pauses inside the clip: the soft breath and decay at pause edges
  are part of what the clone copies.
- **Exact transcript.** The encoder needs the clip's exact words.
- **Speaking style carries over.** Whatever the clip does all the time (a reading cadence, a whisper, a habitual
  sigh), the clone tends to do everywhere. Pick a stretch that sounds like the delivery you want.

Then listen: compare two or three candidate clips on the same texts ([Blind listening](#blind-listening)) and keep the
one you prefer.

### Server side: the engine flags

#### The launch

With the drama-tensorfold launcher (Quickstart in its README), the serve flags go after `--`:

```sh
python3 recipe/pilot.py --weights nvfp4 --runtime-dir runtime --gpu-lock "$GPU_LOCK" --evidence-dir runs/voice-01 \
  -- --voices "$PWD/voices" --default-voice NAME \
     --max-frames 4096 --long-form --long-form-chars 2400 \
     --cfg-gamma 1.5 --cfg-batched \
     --allowed-controls 'concerned,curious,warmly,sighs,small laugh,laughs' \
     --silence-onset-s 3.0 --silence-mid-s 3 \
     --silence-redraws 2 --silence-redraw-plain --silence-redraw-plain-scope controls \
     --short-reply-ratio 0.5 \
     --register-redraw-st 4 --register-redraw-high 20 --register-high-hz HZ
```

`NAME` is your voice file, and `HZ` the high pitch line for your voice (see the `--register-high-hz` row below). Keep
the server on loopback; see the license notes on hosted services in [License and NOTICE](#license-and-notice).

#### Every flag, its value, and why

The server's silence guard is always on in this engine: a synthesis whose frames stay below -40 dBFS for 10 s in a row
ends with finish reason `silence`, its audio kept. The flags below build on it.

| Flag | Value I run | What it prevents | Why this value, and the evidence |
|---|---|---|---|
| `--max-frames` | 4096 | A long reply clipped at the default frame cap (1024 frames, about 47.6 s), finishing `length` | 4096 frames are about 190 s, room for any single reply I send. |
| `--long-form`, `--long-form-chars` | on, 2400 | Joins inside a normal reply | Up to 2400 characters are spoken in ONE generation, so the prosody runs through the whole reply. Longer text is cut at sentence ends into at most 12 balanced segments, each hearing the one before; every join restarts the prosody, and a reply the server speaks in more than one segment also loses the register guard of the rows below: it is sent as drawn, with no register redraw and no register headers, while the silence guards still judge each segment on its own. So I keep replies under the limit. |
| `--cfg-gamma` | 1.5 | Draws drifting toward the model's built-in voice | Classifier-free guidance on the semantic head: each frame's main token is drawn from l_c + (γ - 1)(l_c - l_u), where l_u comes from the same text with no reference. On the reference voice it moved sentence endings toward the reference recording's (significantly at commas), by less than the 1 semitone I had hoped for. γ 1.5 is the value the NVFP4 weights were qualified with; there, first-take failures were 13 of 200 against 33 of 200 for the previous full-size setup without guidance, and none reached a listener in either (two things changed at once, so this is not a clean guidance effect). |
| `--cfg-batched` | on | Guidance doubling the cost of each frame | Decodes the guided and unguided sequences as one batch of two rows, weights read once for both; each row's numbers equal its own unbatched decode. Unbatched, guidance at γ 1.5 to 3 rendered at about 1.2 times real time in my tests. |
| `--allowed-controls` | `concerned, curious, warmly, sighs, small laugh, laughs` | Cues that stall or pull the voice away | Only the six listed controls survive. Any other bracketed span the engine recognizes as a delivery control — roughly, one whose every word is in its shipped cue vocabulary, at least one of them a cue word — goes before any draw reads the text, and so does every tag of the engine's documented scoped set (`whisper`, `emphasis`, `soft`, `fast`, `slow`, `stress`, `forceful`), the text between tags kept. A bracket that is not such a control — a name, a note, a free-form direction whose words fall outside that vocabulary — stays in the text and reaches the model, and so does any other tag; [Delivery cues](#delivery-cues) says what I do about those. Why these six: the evidence in [Delivery cues](#delivery-cues). |
| `--silence-onset-s` | 3.0 | A take that says a word or two and then hums or goes silent, waiting 10 s to be caught | Ends a synthesis once 3 s in a row are below -40 dBFS while it has voiced under 1 s in all. My own replay, over 1,482 draws: 47 of 189 collapses ended at a median of 3.6 s instead of 10.6 s; 8 of 1,293 other draws ended too, each then redrawn. The engine's documentation reports the same trips from its own replay over 1,480 draws (47 of 189 collapses, 8 of 1,291 others); the totals differ from mine, and I cannot account for the difference. |
| `--silence-mid-s` | 3 | A reply that goes silent part-way, then stops with its last sentences never spoken | Makes end of speech inside a run of 3 s or more below -40 dBFS, after 1 s voiced, a `silence`. Replayed over 2,272 draws: it ended 15 of the 2,047 that the 10 s rule left alone; all 8 that had a text to check against had lost words. Holes that recover are left alone, since pacing shortens them anyway. |
| `--silence-redraws` | 2 | A broken take reaching the listener | A take the guard ended (or a short one, below) is drawn again with the next seed, up to twice, each redraw started only if it should finish within the budget. On my spoken-reply texts the redraws caught every first-take failure, so delivered failures were 0. |
| `--silence-redraw-plain` and `--silence-redraw-plain-scope controls` | on, `controls` | A redraw failing the same way as the first take | Redraws speak the text with every delivery control removed: every bracketed span, the documented scoped tags, and a director's-note head — a first line `Direction: …` with `Line:` opening the second — but a note head only counts when the request still has its line breaks. My text step collapses white space ([One request](#one-request)), so the note-head rule never applies to my requests and a note head there would be spoken as words. Cues are where the stalls cluster ([Delivery cues](#delivery-cues)), so the second try is the safest text. |
| `--short-reply-ratio` | 0.5 | A collapse that ends in an end-of-speech token instead of 10 s of silence | A take on 200 characters or more that finishes `stop` with less than 0.5 × characters / 15.6 seconds of audio counts as a trip and is redrawn. 15.6 characters a second is the engine's built-in expected rate; my normal replies sat at a ratio of 0.72 or more. If your voice speaks much faster or slower, check where your normal replies sit before trusting 0.5. |
| `--register-redraw-st` | 4 | A phrase that opens far above the rest, in a thin, high register that sounds stitched in | Draws the whole reply once more when a phrase after the first opens 4 semitones or more above the median of the reply's other phrase openings and enough of it lies above the high line; the draw with the smaller jump is sent. It measures whole replies only: a reply the server speaks in more than one segment (the `--long-form` row above) is sent as drawn, without a register redraw and without the register headers. See the next two rows and [Register and pitch](#register-and-pitch). |
| `--register-redraw-high` | 20 | Tripping on a phrase that only opens high for a moment | At least 20 % of that phrase's voiced frames must lie above the high line. |
| `--register-high-hz` | set from your voice | Tripping on normal emphasis | The engine default is 210 Hz. Set it a few semitones above the median speaking pitch of your reference voice; I calibrated it on my own replies. With 4 semitones and 20 %, the rule flagged 6 of 51 replies from a day's use, including the one a listener had called "stitched"; on 24 test replies it tripped 4, every redraw jumped less, and the 20 untripped replies came back byte for byte unchanged. Those jumping phrases also sounded less like the rest of the reply (speaker similarity 0.729 against 0.786). |

**Deadline header.** Send `X-Deadline-S` with each request: the seconds the client will still wait. A redraw starts
only if the elapsed time plus the last draw's duration fits within the lower of `X-Deadline-S` and the server's
redraw budget (120 s by default), because a stalled draw's length predicts the next one's. I compute the deadline as
20 s + 0.15 s per character, capped at 150 s, and send it less 3 s (kept for the audio steps and the encode), rounded
down to a tenth; a budget under 0.1 s means I do not send at all. The 20 s covers a stall's 10 s guard and connecting;
0.15 s a character is about three renders (the first take and two redraws) at my measured render speed. Measure your
own seconds per character of rendering and set it to about three times that.

**Sampling is fixed on the server.** The sampler is not greedy: temperature 1, nucleus sampling at p 0.9, a 30-token
candidate cut, and a repetition-aware redraw, seeded per request from the server's salt. The server refuses `speed`,
`seed` and `temperature` request fields. The same text on the same server returns the same audio, so a retry from the
client gets the identical take back: every retry has to be a server redraw (seed salt + k), which is why the guards
live in the server.

#### On the full-model engine

The full-model engine (`--weights full`) applies the opt-in patch series listed in the drama-tensorfold README. Its
flags have no defaults and are spelled slightly differently; the mapping of my settings to that engine's flags is:

| My setting | Full-model engine flags |
|---|---|
| built-in guard, -40 dBFS for 10 s | `--silence-floor-dbfs -40 --silence-seconds 10` |
| `--silence-onset-s 3.0` | `--silence-onset-s 3 --silence-onset-voiced-s 1` |
| `--silence-mid-s 3` | `--silence-mid-s 3 --silence-mid-min-voiced-s 1` |
| `--silence-redraws 2`, budget 120 s | `--silence-redraws 2 --silence-redraw-budget-s 120` |
| `--silence-redraw-plain --silence-redraw-plain-scope controls` | `--silence-redraw-controls ''` (an empty list: plain speech) |
| `--short-reply-ratio 0.5` | `--short-reply-ratio 0.5 --short-reply-chars-per-s CPS`, CPS measured on your complete replies |
| `--register-redraw-st 4 --register-redraw-high 20 --register-high-hz HZ` | the same three flag names, and a different detector behind them (see below; the Hz line accepts 75 to 600) |
| `--allowed-controls '…'` | the same |
| `--max-frames 4096` | the same |
| `--long-form`, `--cfg-gamma 1.5 --cfg-batched` | not in that engine; keep replies within one generation's frame budget |

The register row is a flag-name mapping, not a calibrated equivalent. [Register and pitch](#register-and-pitch)
reproduces the register detector my NVFP4 engine's documentation specifies; the full-model patch ships its own
estimator, different in more than precision: pitch is read every 10 ms by normalized cross-correlation over 20 ms
windows ([Register and pitch](#register-and-pitch): 40 ms Hann windows) under Praat's To Pitch (ac) defaults (75 to
600 Hz, voicing 0.45); a frame is kept or dropped against a silence threshold of 0.03 of the take's peak, not against
its speech level; phrases are parted by unvoiced runs of 0.25 s, with a phrase voiced under 0.25 s skipped
([Register and pitch](#register-and-pitch): rests of 0.10 s part phrases, a phrase under 0.5 s joins the one before);
and an opening is the median pitch of a phrase's first 0.25 s voiced ([Register and pitch](#register-and-pitch): its
first quarter, at least 0.4 s). The patch's own calibration instructions apply to that detector: serve without the
register flags, keep whole replies and mark the ones with a stitched-in phrase, and tune trial values against that
server's own register log lines, never against outside measurements. My 4 st, 20 % and HZ were calibrated on the NVFP4
detector; on the full-model engine they are a starting point, not a calibrated setting.

### Text side: writing for the ear and automatic cues

All of this runs on the text before the request, with no model. Drama 3 takes delivery controls as ordinary text, so
where a cue sits, and how a sentence is punctuated, changes what the voice does.

#### Write for the ear

These are the rules I give the writer of a spoken reply (a person or a language model):

| Rule | What it does to the sound |
|---|---|
| Talk it as you would tell a friend; lead with the decision, one idea and one number per sentence, any question last. | Easier to follow on one listen; the key word lands where the voice falls. |
| Keep each sentence to one breath, about 8 to 20 words. | Long run-ons have nowhere to breathe; very short sentences sound clipped. |
| Use fewer full stops: join closely related points with so, and, but, because or which inside one sentence ("the backup finished at six, so the photos are safe"). | Every full stop costs a rest; a reply cut into many short sentences sounds slow and read aloud. |
| No lists, links, symbols, code or "Label: item" lines; one spoken list at most, as "a, b, and c". | In colon-label form the voice ended almost every item like a finished sentence (the recognizer closed 44 of 56 items with a full stop); as one joined list it closed 0 of 12. A cue right before a "Label:" sentence also pulled delivery off-voice. |
| Open a new topic with its subject ("The rename is done"), not with a label. | Same reason. |
| Put a blank line (or the marker in [Topic-shift detection](#topic-shift-detection)) between topics. | It becomes the 1.1 s thinking pause ([Thinking pauses at topic shifts](#thinking-pauses-at-topic-shifts-11-s)). |
| Never open a sentence with a lone filler and a comma ("So,", "Well,", "I mean,", "Okay,"), and never write um, uh, like or you know. | Punctuation sets filler timing: a comma after a filler leaves a gap after it, and a filler right after a full stop gets the long sentence rest before it (0.51-0.56 s against 0.10 s inside a sentence). Sentence-initial fillers also came out stressed or swallowed. A connective inside the clause, with no comma after it, flows. |
| Use the cues in [Delivery cues](#delivery-cues) between sentences only, never on the first or last sentence. | See [Delivery cues](#delivery-cues) and [Drop cues that open a topic or close the reply](#drop-cues-that-open-a-topic-or-close-the-reply). |

If the writer is a language model, check that it keeps following the rules: one of my regressions ("no cues, too fast,
no pauses") was the writer drifting into cue-free, unmarked text while the voice stack was unchanged. Cue-free replies
spoke about 7 % faster on the reference voice. The automatic steps below are the floor for exactly that case.

#### Delivery cues

**Which cues.** The server allowlist keeps six point cues: `[concerned]`, `[curious]`, `[warmly]`, `[sighs]`,
`[small laugh]` and `[laughs]`. What I measured on the reference voice (holes = a quiet run of 3 s or more at the
cue's sentence end):

| Cue class | Evidence | Decision |
|---|---|---|
| `[sighs]`, `[small laugh]`, `[laughs]` | Holes 0/66, 2/61 and 1/37, against 5/493 sentence ends with no cue. On my renders `[sighs]` gave a longer rest with an audible breath. | allowed |
| `[concerned]`, `[curious]`, `[warmly]` | Placed by the policy below on 20 paired takes: no extra redraws (McNemar p = 1), speaker similarity 0.773 → 0.778. | allowed |
| `[pause]`, `[hesitates]` | Holes 9/69 and 5/47, one with 8.45 s of silence and then `stop`. Taking both out cut first-take trips from 10/62 to 0/62 (McNemar p = 0.002). | removed by the allowlist |
| Breath tags (`[shaky breath]`, `[takes a breath]`), `[gasps]`, `[clears throat]` | A beat with no breath or gasp sound; `[clears throat]` was spoken as "Mm-hmm". | removed |
| Style prefixes in the engine's cue vocabulary (`[pleased]`, `[softly, whispering]`) | More range in a denser probe, but endings that fall further (stop -1.0 st, last phrase -1.6 st); `[pleased]` before a label made the voice skip words and jump about 6 semitones. | removed: every word is a vocabulary cue, so the allowlist takes them |
| Scoped tags of the engine's documented set (`<soft>`, `<whisper>`) | A quiet run of 5.9 s after `<soft>` | removed, the text between the tags kept |
| Free-form directions (`[barely holding it together]`, "speak quietly"), brackets and tags outside the two sets above | A "speak quietly" direction on the built-in voice collapsed into a hum | not removed: they are not delivery controls, so the first draw sends them to the model (only a plain redraw, [Every flag, its value, and why](#every-flag-its-value-and-why), drops bracketed ones). The only guard is that I never write them, and the automatic step adds none |

**How many.** About one cue per 120 characters of speech text, roughly one every 7 seconds at a typical pace; none
in a reply under 120 characters. At this density the cued replies gained range (pitch range 8.25 → 9.20 st, p 0.067;
pitch movement 12.93 → 13.80 st/s, p 0.19; 20 pairs) with speaker similarity unchanged. The policy placed fewer cues
than its budget allows (25 over ten replies); in short probes with one point cue each, about one per 240 characters,
point cues gave a significant +1.17 st of range and +1.42 st/s of movement (p 0.027) without moving the endings. An
early reply with no cues at all was heard as robotic.

**Where.** Mid-topic only; never on the reply's first sentence, never on the first sentence of a topic, never on the
last sentence, never on two neighbouring sentences.

**Placing them automatically (only when the writer put none in).** A reply that already holds any of the six cues
goes out as written: the writer's cues win. Otherwise:

1. Split the reply into sentences ([Topic-shift detection](#topic-shift-detection)'s rules). With fewer than three
   sentences, add nothing.
2. Read each sentence's words, curly quotes as straight ones, whole words only, case-insensitive, never inside a
   hyphenated compound ("pass-or-fail" does not hold "fail"). A word inside quotes never causes a cue.
3. Give each sentence its trigger, first match wins:

   | Cue | Trigger | Not when |
   |---|---|---|
   | `[sighs]` | a first-person slip: I missed, I've missed, my mistake, my fault, I got that/it/this wrong, I should have, I forgot, I messed up, that's on me, that was on me, I'm sorry, I am sorry, sorry about that, "sorry, I" | the sentence holds a negation (not, never, no longer, without, unfixed, any n't), says "the word", or offers condolence (sorry for your, sorry to hear, sorry about your); at most once per reply |
   | `[concerned]` | a problem word: fail(s/ed/ing/ure/ures), error(s), bug(s), crash(es/ed/ing), broke, broken, problem(s), issue(s), risk(s), risky, deadline(s), overdue, expires/expired/expiring, missing, lost, leak(s/ed/ing), outage(s), warning(s), urgent, stuck, blocked, stall(s/ed/ing), timeout(s), time out, timed out, breach(es/ed), exposed, vulnerable, over budget, went over, behind schedule, nearly full, out of space, down for, regression(s), regressed, slipped, corrupt(ed), unfixed, died; or a fix reported as not done (a negation up to two words before fixed, resolved or solved) | the problem word sits inside a phrase ruling it out (no errors/issues/problems/failures/bugs/risks, nothing lost or broken, without any errors/issues/problems/loss, zero errors/failures, all clear), or the sentence holds a fix word (fixed, resolved, solved, no longer) |
   | `[curious]` | the sentence asks the listener something: it ends with "?" or opens (after an optional So, And or Now) with do you want, should I, can you, could you, would you like, would you rather, want me to | it holds a negation; and only on the second or third sentence (a later `[curious]`-led sentence reads as a new topic in [Topic-shift detection](#topic-shift-detection), and would then be dropped by [the drop rule](#drop-cues-that-open-a-topic-or-close-the-reply)) |
   | `[small laugh]` | the sentence after one with a wry marker: luckily, turns out, turned out, of course, ironically, funnily enough, as luck would have it | the marked sentence, the cued one or either neighbour holds a problem word or an unfinished fix; the marked or cued sentence holds a negation or contrast; at most once |
   | `[warmly]` | a positive, thanks or done sentence: thanks, thank you, is/are/was/it's/that's/all/now/got done, finished, all set, good news, great news, works, working again, passed, ready, glad, nice work, well done, landed, shipped, merged, succeeded, complete, completed, approved, fixed, resolved, sorted | it holds a negation or a contrast (but, however, instead, though, although, still, stopped, waiting, no, except, only, yet), or it or a neighbour holds a problem word or an unfinished fix |
   | `[laughs]` | never placed automatically: whether something is funny cannot be read from the words | |

4. Skip the first sentence, the last, and every topic opener (after a blank line, after the marker, or where
   [Topic-shift detection](#topic-shift-detection) would mark one).
5. Budget: floor(characters of speech text / 120) cues.
6. Fill the budget one cue class at a time in this rank: sighs, concerned, curious, small laugh, warmly; within a
   class, earlier sentences first. Skip a sentence next to one already cued.
7. Insert `[cue] ` at the start of each chosen sentence. Removing the inserted tokens must give the original text
   back byte for byte.

These are word lists, not understanding. A warning phrased without any listed word ("the queue has been growing all
afternoon" matches nothing) gets no `[concerned]`, and a sentence can hold a listed positive word beside an unlisted
problem ("the vendor fixed the login by breaking the export": "fixed" triggers, "breaking" is not a problem word, so
the `[warmly]` veto never fires). They lower the risk of a wrong cue; they do not remove it. When in doubt the rule
adds nothing.

#### Topic-shift detection

The thinking pause ([Thinking pauses at topic shifts](#thinking-pauses-at-topic-shifts-11-s)) needs to know where the
subject changes. If the writer marked topics (blank lines, or a marker such as `¶` when the transport flattens
newlines), use those. Otherwise mark them with these rules, in this order; when unsure, mark nothing. A reply with
fewer than three sentences gets no marks.

| Reason | A shift follows sentence i when |
|---|---|
| paragraph | a blank line sits between sentence i and i + 1 |
| line | a single line break sits there, counted only when such breaks sit at no more than half the boundaries (a writer who breaks every line is formatting, not marking topics) |
| label | sentence i + 1 opens with a topic label, written ("Garden: …", up to four words and a colon) or spoken ("On the backups side, …"), and the reply has two or more such labels |
| move | i + 1 opens with Okay, Now, Anyway, Next, Finally, Also, Meanwhile, Separately, Lastly or One more thing (each followed by a comma), or with Here's, Here is, One thing, Two things, One more thing, The other one, Last thing |
| stance | i + 1 opens with Honestly, Frankly, To be fair or On another note (with a comma), with two or more sentences before it |
| plan | i + 1 opens with The plan:, The plan now:, The plan from here:, The plan for tonight: or Next steps:, with two or more sentences before it |
| frame | i is the first sentence and only frames the reply in eight words or fewer (Here's…, Quick…, A quick…, A short…, "Okay, the…", or a summary noun such as "Evening status, short version."), and the reply has four or more sentences |
| ask | i + 1 is led by a `[curious]` cue with three or more sentences before it; or i + 1 is the last sentence, a question or a request (tell me, let me know, your call, up to you), in a reply of four or more sentences that already has another shift |
| closing | i + 1 is the last sentence and signs off (nothing else, that's all, that's it for now, that is all, I'll report back, talk soon, more tomorrow, good night, or a `[warmly]` lead), in a reply of four or more sentences that already has another shift |

"So" and "But" openers and changes of tone are left alone. Sentence boundaries: a run of `.` `?` `!` or `…`, optional
closing quotes or brackets, white space, then a capital, digit, opening quote or bracket. Cue spans `[…]` and tag
spans `<…>` are opaque. Common abbreviations (Mr, Mrs, Dr, vs, etc, e.g, i.e, a single letter, "No" before a digit)
do not end a sentence. A piece that holds only cues belongs to the sentence before it.

#### Drop cues that open a topic or close the reply

After marking shifts, and whoever wrote the cues:

- **Drop every cue that opens a topic** (right after a shift). On the reference voice, a cue placed there pulled the
  first sentence off the voice.
- **Drop every cue in the last sentence, or on its own line right before it.** A cue there lifted the sign-off's
  pitch, so the ending sounded stitched on. This rule needs sentence structure, so it lives in the text step, not in
  the server, which cannot tell a sign-off from any other sentence.

#### One request

Turn each topic mark into a paragraph break for the pause step, then collapse all white space to single spaces for
the request, so the server never sees (or speaks) the marker. That collapse has a consequence: the server's plain
redraw ([Every flag, its value, and why](#every-flag-its-value-and-why)) drops a director's-note head — a first line
`Direction: …` with `Line:` opening the second — only from requests that keep their line breaks, and one-line text
has no such head, so a note head in my text would be spoken as words in every draw. Nothing in this pipeline writes
note heads, so nothing is lost; the collapse is still a choice with that cost, not a free simplification. Send the
whole reply in one request, with `"response_format": "wav"` and the deadline header
([Every flag, its value, and why](#every-flag-its-value-and-why)). Splitting a reply into several requests restarts
the prosody at every join.

### Delivery: only a finished take is sent

Use the audio only when the response header `X-Finish-Reason` is `stop`. `length` means the frame cap clipped it;
`silence` means the guard ended it and every redraw the budget allowed failed too. In both cases, and when the server
is busy, misses the deadline or returns broken audio, I send no audio at all and let the text reply stand. A `stop`
does not by itself prove the words were spoken (a take once stopped speaking after 6.5 s and padded about 64 s of
silent frames before ending), which is what the onset, mid and short-reply rules and the measurements in
[How to measure your own voice](#how-to-measure-your-own-voice) are for. I log, per reply, the server's redraw counts
(`X-Silence-Redraws`, `X-Register-Trip`, `X-Register-Redraws`, `X-Silence-Mid-Trips`) — the two register headers are
sent only on a whole single-segment reply, so my logging takes them as absent when the server sends none — the cues
found, added and dropped, and the pauses placed and skipped, and never the text itself.

### Audio side: thinking pauses, breaths, pacing, encoding

The server returns 16-bit mono WAV at 44.1 kHz. Everything below works on those samples with numpy. Speech samples are
never changed: only rests get longer or shorter.

#### Finding rests

A **rest** is a run of 10 ms frames whose RMS is below **-42 dBFS**, between speech (not at the very start or end),
lasting **at least 0.15 s**.

#### Matching sentence ends to rests

To lengthen the rest at a sentence end you need to know which rest that is, without a forced aligner:

1. Split the text (cues removed) into sentences; count each sentence's syllables as vowel groups (`[aeiouy]+`, each
   number counted as one).
2. Find the take's rests ([Finding rests](#finding-rests)), where speech starts and where it ends.
3. Match sentence ends to rests in order by dynamic programming. For each stretch between two consecutive matches,
   score -(ln(speech time / expected time) / 0.35)², where expected time is the stretch's share of the reply's
   syllables times the total speech time; add 1.0 × ln(rest length / 0.15 s) for every rest used; subtract 2.0 for
   every sentence end left unmatched, and never leave more than 3 in a row unmatched. Take the best path.
4. A sentence end no rest fits stays unmatched; a pause there is skipped, never guessed.
5. Refuse the whole take (place no pauses) when the text and audio do not belong together: syllables per second of
   speech outside 4.5 to 13, or a root-mean-square log ratio of the matched stretches over 0.6.

I tuned the 0.35, 1.0 and 2.0 on 118 takes and checked them on 22 others.

#### Thinking pauses at topic shifts: 1.1 s

At each topic shift, lengthen the matched rest to **1.1 s** (a rest already that long is left as it is). Without
this, topics ran together: the pacing cap held every sentence rest near 0.55 s, and the rests at shifts had a median
of 0.53 s, one as short as 0.06 s. With 1.1 s pauses the listener judged the replies much better, and every word
outside the inserted fill stayed bit-identical. The pauses did not change the pitch within topics (pitch spread
2.66 → 2.70 st, p ≥ 0.52). 1.1 s is a listening preference (0.85 s was the other candidate), not an imitation of any
recording's natural rests.

I render the reply in one piece and insert the pauses afterwards. Rendering each topic separately and joining the
audio restarted the prosody at every join, and an engine-side variant that split the render at the marks was slower
(28.6 s → 35.6 s of render time over 12 paired takes).

#### Breaths inside a topic: 0.85 s

Walking the sentence ends from the start of the reply, a **breath** goes after a sentence end once **2 or more
sentences and 200 or more characters** of speech (cues left out) have passed since the start, the last shift or the
last breath; never at a shift, at the sentence end right before one, or at the reply's end. Its rest is lengthened to
**0.85 s** by the same rules as a shift. In short replies this rarely fires (most of my topics were one to three
sentences, and a topic needs four to get a breath); it matters in long stretches about one subject.

#### Filling a lengthened rest

1. Split the rest at its centre and insert the extra samples there; the breath and word tails at the rest's edges are
   untouched.
2. **Room tone:** among the take's rests, look for 0.20 s windows of whole 10 ms frames at least 0.05 s clear of each
   rest's edges, with every frame above -90 dBFS (not digital silence), below -42 dBFS, and all within 6 dB of each
   other. Take the window whose level is closest to the rest being lengthened, and loop it forwards then backwards.
3. **No clean room tone:** use Gaussian noise at **-65 dBFS**, seeded by the position, so a reply never goes dead
   silent.
4. Join with **15 ms** crossfades. The fill spans the inserted samples plus 15 ms at each end: its first 15 ms
   crossfades with the take's last 15 ms before the cut, its last 15 ms with the take's first 15 ms after it. Each
   crossfade outputs, per sample, (a·cos θ + b·sin θ) / sqrt(1 + 2ρ·cos θ·sin θ), where `a` is the signal before the
   join fading out, `b` the signal after it fading in, θ sweeps linearly from 0 to π/2 across the 15 ms, and ρ is the
   two 15 ms windows' correlation a·b / sqrt((a·a)·(b·b)), clipped to 0-1 and 0 when either window is silent. Equal
   power for unrelated signals (ρ 0), equal gain for the same tone (ρ 1), so a loop of the take's own tone does not
   bump 3 dB.
5. **Check:** measured on the result, the lengthened rest must still be one run of frames under -42 dBFS. If not,
   skip that pause and render again without it.

#### Pacing

Then pace the whole take, cutting on the same 10 ms grid:

| Setting | Value | Why |
|---|---|---|
| rest floor | -42 dBFS | the same rest definition as [Finding rests](#finding-rests), so pacing and pauses agree |
| rest cap | 0.55 s | a longer rest is cut to 0.55 s from its centre, so the breath at either edge survives; this tightens the model's occasional long gaps without touching speech |
| lead | 0.10 s | leading silence trimmed to 0.10 s, so the voice starts promptly |
| tail | 0.20 s | trailing silence trimmed to 0.20 s |
| shift pause | kept up to 1.1 + 0.6 s | a rest overlapping a placed shift pause is exempt from the cap up to 1.7 s, a safety limit in case a stall lands there |
| breath | kept up to max(0.85, 0.55) s | a rest overlapping a placed breath keeps 0.85 s, so a long rest left alone there still comes out as a breath |

Pacing changed the length of a take by about 1 %, so it does not speed the voice up; the pace comes from the text
([Write for the ear](#write-for-the-ear)) and the reference clip ([Choosing a reference clip](#choosing-a-reference-clip)).

#### Encoding

One lossy encode at the end, never two:

| Format | Settings | Why |
|---|---|---|
| Ogg/Opus | mono, 64 kbit/s, VBR, `-application audio`, `-compression_level 10` | Common 48 kbit/s "voip" voice-note settings cut 4-8 kHz by 2-5 dB on my renders and added distortion above 4 kHz; 64 kbit/s in audio mode was transparent. An earlier lossy voice-note encode dulled the audio and added buzz. |
| MP3, for clients that only play MP3 | mono, 96 kbit/s CBR, LAME quality 2, at the take's sample rate | the plain choice for such clients; I have no measured comparison of other MP3 rates |

The ffmpeg arguments for the Opus encode:

```sh
ffmpeg -f s16le -ar 44100 -ac 1 -i paced.pcm -acodec libopus -ac 1 -b:a 64k -vbr on \
  -application audio -compression_level 10 -f ogg reply.ogg
```

### What did not work

| Tried | What happened | Why I think so |
|---|---|---|
| A sped-up reference clip, for pace | Faster, but the breathiness went | Speeding the clip changes the voice the model copies; fewer full stops in the text ([Write for the ear](#write-for-the-ear)) gave pace and kept the breaths |
| Time-stretching the finished audio slower (pitch kept, tempo ×0.85) | It sounded electronic | My reading, not measured: stretching smears consonants and breaths. I take pace from the text instead |
| Voice-side levers for sentence-final intonation | No reference choice or order, punctuation change (dropping the final full stop, joining the last two sentences) or audio processing levelled the clone's falling endings | The endings come from the model's prosody; these levers do not reach it |
| Classifier-free guidance as a fidelity lever | γ 1.5 or 2 did not raise held-out speaker match | It helps endings a little ([Every flag, its value, and why](#every-flag-its-value-and-why)), not timbre |
| EQ or de-buzz post-filters | Not used: with byte-identical prompts the reference implementation sounded no warmer than this engine | The "buzzy" sound was not an engine defect; the levers are wording, the clip and listening |
| `[pause]` and `[hesitates]` for pauses | Holes of 3 s or more, up to 8.45 s of silence | Pause-like tags are the stall class; pauses are made in the audio step instead ([Thinking pauses at topic shifts](#thinking-pauses-at-topic-shifts-11-s)) |
| Style prefixes and free-form directions | More range, but endings fell further; one direction collapsed into a hum | See [Delivery cues](#delivery-cues) |
| Inserted fillers ("So,", "Well,", "I mean,", "and yeah") | Heard as unnatural; fillers narrowed the range (8.76 → 7.98 st, p 0.016); every comma form left a gap after the filler; sentence-initial fillers came out 4-5 dB stressed or about 20 dB swallowed | The model reads a filler as a word to deliver, not as glue; connectives belong inside the clause, written by the writer ([Write for the ear](#write-for-the-ear)) |
| Many short sentences | Slow, read-aloud delivery | Each full stop costs a rest |
| A cue right before the closing sentence | The sign-off's pitch jumped and sounded stitched on | Hence the drop rule ([Drop cues that open a topic or close the reply](#drop-cues-that-open-a-topic-or-close-the-reply)) |
| A retry from the client | The identical take came back | The server is deterministic per text; redraws must be server-side ([Every flag, its value, and why](#every-flag-its-value-and-why)) |
| Splitting a reply into separately rendered segments | Prosody restarted at every join | Render one piece; insert pauses afterwards ([Thinking pauses at topic shifts](#thinking-pauses-at-topic-shifts-11-s)) |

### How to measure your own voice

Measure every change on paired takes (same texts, same seeds, before and after), over many texts, and listen.

#### Word error rate

Transcribe each delivered take with a speech recognizer and score it against the text with cues and markers removed.
A take with a word error rate over 0.2 has lost words; on the reference voice, normal takes sat around 1-2 % (1.31 %
over 96 takes for the NVFP4 weights). Also check that the text's last three words appear in the last fifteen heard:
lost endings hide in an otherwise good score.

#### Speaker similarity

Embed each take and the reference clip with a speaker encoder (ECAPA) and take the cosine similarity. Read it as
paired differences, not absolute values: my cued replies scored 0.773 → 0.778 against uncued ones, and I treat a drop
of more than 0.02 as a regression. A single phrase that sounds unlike the rest shows up as a lower similarity of that
phrase to the rest of its reply (0.729 against 0.786 for the jumping phrases in
[Every flag, its value, and why](#every-flag-its-value-and-why)).

#### Register and pitch

The register measure, numpy only:

1. Pitch: autocorrelation on 40 ms Hann frames every 10 ms at 16 kHz, lags for 70-400 Hz. A frame is voiced when its
   normalised peak is 0.45 or more and it is within 25 dB of the speech level; a half-lag peak scoring 0.9 of the best
   wins, which avoids sub-octave picks. (Against pYIN: median frame difference 0.14 st.) The speech level is the mean
   power of the frames within 20 dB of the 95th percentile: of the 10 ms frames' dB values, take every frame at or
   above the 95th percentile minus 20 dB and average those frames' power, in dB. This is the estimator my NVFP4
   engine's documentation specifies for its register measure, so the server's trips and your measurements count the
   same frames as speech — with the engine's own limit on that claim: its arithmetic differs from numpy's in the
   last bits and its rounding at exact ties, so a value sitting right on a threshold can fall the other way. (The
   full-model engine measures register with a different estimator of its own; see
   [On the full-model engine](#on-the-full-model-engine).)
2. Phrases: speech between rests of 0.10 s or more (frames 25 dB under the speech level); a phrase under 0.5 s joins
   the one before. A 0.30 s rest missed a real jump that came with 0.2 s from word to word.
3. Per phrase after the first: its opening (median pitch of its first quarter, at least 0.4 s) minus the median of the
   other phrases' openings, in semitones; and its share of voiced frames over your high line.
4. Flag a reply when a phrase opens 4 st or more above the others with 20 % or more over the line. Raw "opens 3 st
   above the reply median" is common and means little; the excess over the other openings is what separates a jump.

#### Pitch movement and range

On each delivered take, as you deliver it (after pacing): pitch spread (standard deviation, st), 5-95 % range (st),
mean phrase velocity (st/s), the pitch fall at commas, at full stops and on the last phrase, energy range, and rate
variation. Energy range is the 5th-to-95th-percentile spread of the dB of the speech frames (the 10 ms frames within
25 dB of the speech level, [Register and pitch](#register-and-pitch)); rate variation is the standard deviation, over
the reply's phrases, of syllables a second, a syllable counted acoustically as a voiced peak of the frame level smoothed over 50 ms with 2 dB of prominence. On
the reference voice, a reply a listener had heard as flat measured a spread of 1.62 st and a range of 5.0 st, and the
same reply with the cue policy applied 3.20 and 8.7, with velocity 10.2 → 14.0 st/s, at a cost: speaker similarity
0.783 → 0.753
and a full-stop fall of -1.25 → -4.86 st. One draw each, so treat it as an illustration. Over 20 pairs
the cue policy gave range 8.25 → 9.20 st. More range usually costs endings that fall further: watch both together.

#### Pause statistics

From the delivered take: the rest at each topic shift (it should be about 1.1 s), at each breath point, and at other
sentence ends (mostly at the 0.55 s cap); the longest quiet run anywhere (3 s or more inside speech is a hole); and the
time to first speech. From the server headers and your log: how many shifts and breaths were placed and why the rest
were skipped.

#### Are the cues performed?

A metric that scores only words and speaker match will rate a flatter take as fine. In one comparison, the newer
setup dropped the scripted sigh and laugh and the pauses, moved its pitch about a third less, and left two one-second
silences where the cues were; word error and similarity did not notice, and a listener did. So for every cued
sentence check, against the same sentence uncued: the rest or breath at the cue (`[sighs]` should leave a longer rest
with audible breath; on my renders `[laughs]` left a beat of about +0.47 s and `[small laugh]` about +0.31 s),
the pitch movement of the cued sentence, and no hole of 1 s or more at the cue. And listen to the cued sentences.

#### First-take failures, retakes and timing

Count a first-take failure when the take did not finish `stop`, was redrawn (`X-Silence-Redraws` above 0), has a word
error rate over 0.2, or holds a quiet run of 3 s or more. Count delivered failures separately (they should be 0).
When you time replies, count the redraws too: a redraw costs a whole render, so a wait figure without the retake count
hides the slow cases. Count register redraws apart from silence redraws; they are not failures.

#### Blind listening

Build a pack of the same texts in each version, labels hidden and order shuffled, levels matched (I use -23 dBFS for
takes), and let the listener pick. Many of the decisions here (1.1 s pauses, the reference clip, dropping fillers)
were made this way, after the metrics had said what changed but not whether it was better.

### Reproduction checklist

1. Download the weights under the license and build the engine ([Quick start](#-quick-start)).
2. Encode a clean reference clip of a voice you have the rights to use
   ([Choosing a reference clip](#choosing-a-reference-clip)) into the voice directory your launch will pass as
   `--voices`.
3. Start the server with the flags in [The launch](#the-launch) — it reads the voice directory once, at startup, so
   the voice files must be there first — then set `--register-high-hz` from your reference voice and check
   `--short-reply-ratio` against your normal replies.
4. Give the writer the rules in [Write for the ear](#write-for-the-ear).
5. Implement the cue, topic-shift, drop and one-request rules
   ([Text side](#text-side-writing-for-the-ear-and-automatic-cues)), send one request with `X-Deadline-S`, and keep
   only `stop` takes ([Delivery](#delivery-only-a-finished-take-is-sent)).
6. Implement the [Audio side](#audio-side-thinking-pauses-breaths-pacing-encoding) steps on the audio.
7. Measure ([How to measure your own voice](#how-to-measure-your-own-voice)) on paired texts and seeds, and listen
   blind before you keep any change.

### Honest caveats

**About the numbers.** Every measurement is from my own renders: most "on the reference voice" (the one cloned voice I
use now), some on earlier voices or the built-in voice, spoken replies of a few sentences to a few minutes, the NVFP4
pruned Drama 3 weights on one GB10 unless I say otherwise. They show what each step does and in which direction; they
are not promises for your voice. Drama 3 draws are random, and two draws of the same text can differ a lot (on one
reply, pitch spread was 1.62 semitones on one draw and 2.74 on another), so judge any change over many paired texts
and seeds, never one take.

Around this pipeline sit two kinds of work I am deliberately not describing: the pieces that tie the voice to one
client (that client's fixed wait, its one-request-at-a-time handling, and the tooling that saves each take for
measurement), and experiments specific to one particular reference clip. Neither is needed to reproduce the voice;
every technique that shapes it is in the sections from [Server side](#server-side-the-engine-flags) to
[How to measure your own voice](#how-to-measure-your-own-voice).

### License and NOTICE

This document is licensed under the Apache License, Version 2.0; see the LICENSE file in this repository. The
[drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold) engine it builds on is an upstream dependency
with its own license and notices, not part of this repository; the [NOTICE](NOTICE) file carries the notices that
must travel with the Fish Audio materials and credits TensorFold.

**Built with Fish Audio.** The Drama 3 model weights and codec checkpoint, and the NVFP4 pruned derivative, are
licensed under the
[Fish Audio Community License, Version 1.0](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE),
not under this repository's license. They are not in this repository and are not redistributed by it: you download
them yourself from Hugging Face. Their terms apply to them and to their outputs, including the limits on commercial
use and the need for a separate license for a hosted service. Anyone who redistributes those files must keep this
notice with them, as that license's Section IV(a) requires:

> These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC.
> All Rights Reserved.

Clone only a voice you have the rights to use: your own, one whose owner has given consent on record, or one the model
invented. The license forbids speech that imitates an identifiable person's voice without their prior consent, and
impersonation to deceive. This is a summary, not legal advice: read the license.

---

[License and NOTICE](#license-and-notice) · [NOTICE](NOTICE) · [LICENSE](LICENSE) · [drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold)
