# korean-tts-pipeline

English-to-Korean speech translation pipeline with personalized TTS. Combines speech recognition, machine translation, and a FastSpeech2-based TTS model fine-tuned on personal voice data.

## Overview

This project integrates three independent models into a single end-to-end pipeline: STT (speech recognition), translation, and TTS (speech synthesis). The TTS model was fine-tuned on personal voice data on top of a provided FastSpeech2 base implementation, enabling personalized speech synthesis beyond the pretrained model.

## Pipeline

```
English speech input (input.wav)
  |
  |--> [STT] Wav2Vec2 (facebook/wav2vec2-base-960h)
  |       -> English text
  |
  |--> [Translation] MBart50 (facebook/mbart-large-50-many-to-many-mmt)
  |       -> English to Korean
  |
  |--> [TTS] FastSpeech2 + VocGAN
          -> Korean speech synthesized in a personally fine-tuned voice
```

## Key Contributions

- **Pipeline integration** (`speech_translation_pipeline.py`) — connects STT, translation, and TTS into a single runnable flow
- **Personalized TTS fine-tuning** — fine-tuned the base FastSpeech2 model on personal voice data for 313 epochs (230k steps), producing synthesis results in a personal voice rather than a generic pretrained voice
- **Korean text preprocessing** (`preprocess.py`, `synthesize.py`) — G2P (grapheme-to-phoneme) conversion, TextGrid alignment, mel/f0/energy feature extraction

## Usage

```
python speech_translation_pipeline.py
```

Given `sample_input.wav` (English speech) as input, the pipeline generates a translated and synthesized Korean wav file in `./results/`.

## Results

- Fine-tuned synthesis sample: `examples/sample_output.wav`
- Fine-tuning loss curve (313 epoch, 230k step): `examples/loss_curve_finetuned.png`
- Pretrained model loss curve (80k~350k step): `examples/loss_curve_pretrained.png`
- Synthesized spectrogram: `examples/synthesized_spectrogram.png`

## Models Used

- STT: Wav2Vec2 (Hugging Face `facebook/wav2vec2-base-960h`)
- Translation: MBart50 (Hugging Face `facebook/mbart-large-50-many-to-many-mmt`)
- TTS: FastSpeech2 + VocGAN (KSS dataset, fine-tuned on personal voice)

## Environment

- PyTorch, torchaudio
- transformers (Hugging Face)
- g2pK, jamo (Korean text preprocessing)
