# Voice Studio

A desktop app that turns a song into an AI vocal backing track sung in the user's own voice, and trains the voice models that make it possible, entirely on the local PC.

## Language

### Songs and stems

**Song**:
The original mixed track the user wants to cover.
_Avoid_: Target, input track

**Acapella**:
The original singer's isolated vocals, split out of a Song.
_Avoid_: Original vocals, vocal stem, source vocals

**Clean Acapella**:
An Acapella with reverb and backing harmonies removed, ready for Conversion.
_Avoid_: Dry vocals, processed vocals

**Instrumental**:
The Song with the vocals removed.
_Avoid_: No vocals, backing track, karaoke

**AI Vocals**:
The Clean Acapella re-sung in the user's voice by a Voice Model. This is the 垫音 track played under the live mic.
_Avoid_: Converted vocals, output, cover vocals

### Voices

**Voice Model**:
A trained RVC model of one person's voice: a `.pth` checkpoint plus its matching `.index` file.
_Avoid_: Model (alone), checkpoint, voice

**Voice Dataset**:
Dry, noise-free recordings of one person speaking and singing, used to train a Voice Model.
_Avoid_: Training data, samples

**Pitch Shift**:
The number of semitones applied during Conversion; a multiple of 12 keeps AI Vocals in key with the Instrumental.
_Avoid_: Transpose, key change

### Work

**Cover Job**:
One Song taken through Separation, Cleanup and Conversion to produce its Instrumental and AI Vocals.
_Avoid_: Task, run, render

**Training Job**:
One Voice Dataset taken through preprocessing, feature extraction, training and indexing to produce a Voice Model.
_Avoid_: Fine-tune, training run

**Separation**:
Splitting a Song into Acapella and Instrumental.
_Avoid_: Demixing, stem split

**Cleanup**:
Removing reverb and backing harmonies from an Acapella.
_Avoid_: De-reverb (alone), denoise

**Conversion**:
Re-singing a Clean Acapella with a Voice Model.
_Avoid_: Inference, voice changing

**Breath Restoration**:
Replacing the pitchless stretches of AI Vocals (breaths, s/sh sounds) with the same stretches of the Clean Acapella, matched to the AI Vocals' level.
_Avoid_: Breath fix, protect (that's an RVC setting)

**Preview Mix**:
Instrumental and AI Vocals played together in the app at stream volumes, before going live.
_Avoid_: Mixdown, render
