import os
import torch
import torchaudio

# -----------------------------
# STT 모델
# -----------------------------
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# -----------------------------
# MBART 번역 모델
# -----------------------------
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# -----------------------------
# FastSpeech2 + VocGAN
# -----------------------------
import hparams as hp
from synthesize import get_FastSpeech2, kor_preprocess, synthesize as fs2_synthesize
from utils import get_vocgan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔹 STT / 번역 모델 로드 중...")

# STT
STT_MODEL_NAME = "facebook/wav2vec2-base-960h"
stt_processor = Wav2Vec2Processor.from_pretrained(STT_MODEL_NAME)
stt_model = Wav2Vec2ForCTC.from_pretrained(STT_MODEL_NAME).to(device)

# 번역 (MBART)
MT_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL_NAME)
mt_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL_NAME).to(device)


# ------------------------------------------------
# 1. Speech → English text
# ------------------------------------------------
def speech_to_text(audio_path: str) -> str:
    print(f"\n[1] STT 실행: {audio_path}")

    speech, sr = torchaudio.load(audio_path)

    if speech.size(0) > 1:
        speech = speech.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        speech = resampler(speech)
        sr = target_sr

    inputs = stt_processor(
        speech.squeeze(0),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = stt_model(inputs.input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    text = stt_processor.batch_decode(predicted_ids)[0]
    text = text.strip()
    print("  ➡ STT 결과(영어):", text)
    return text

# ------------------------------------------------
# 2. English → Korean 
# ------------------------------------------------
def translate_to_korean(english_text):
    print("🌐 [Translate] 영어 → 한국어 번역 중...")

    mt_tokenizer.src_lang = "en_XX"

    encoded = mt_tokenizer(english_text, return_tensors="pt").to(device)  

    generated_tokens = mt_model.generate(
        **encoded,
        forced_bos_token_id=mt_tokenizer.lang_code_to_id["ko_KR"]
    )

    korean_text = mt_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("➡ 번역 결과:", korean_text)
    return korean_text



# -----------------------------
# 3. Korean text → Speech (FastSpeech2 + VocGAN)
# -----------------------------
def tts_korean(korean_text: str):
    print("\n[3] TTS 실행 (한국어 텍스트 → 음성)")

    # FastSpeech2 모델 로드 (사전학습 checkpoint step 사용)
    step = 350000  # checkpoint_350000.pth.tar
    print(f"  ▸ FastSpeech2 checkpoint step = {step}")
    model = get_FastSpeech2(step)
    model = model.to(device)
    model.eval()

    # VocGAN 로드
    vocoder = get_vocgan(hp.vocoder_pretrained_model_path)

    # 한글 전처리 + 텍스트 텐서 생성
    text_tensor, src_len = kor_preprocess(korean_text)  

    # 기본값 설정 (None 처리 방지)
    dur_pitch_energy_aug = [1.0, 1.0, 1.0]  

    # 음성 합성 (파일은 ./results/ 아래에 저장됨)
    fs2_synthesize(
        model,
        vocoder,
        text_tensor,
        korean_text,
        prefix="pipeline",  
        dur_pitch_energy_aug=dur_pitch_energy_aug,  
        src_len=src_len  
    )
    print("  ➡ 합성 완료! ./results 폴더에 wav 파일이 생성됩니다.")


# ------------------------------------------------
# 4. Full PIPELINE
# ------------------------------------------------
def run_pipeline(input_wav="input.wav"):
    print("\n🚀 전체 파이프라인 시작\n")

    english = speech_to_text(input_wav)
    korean = translate_to_korean(english)
    tts_korean(korean)

    print("\n✅ 파이프라인 완료")


if __name__ == "__main__":
    run_pipeline()
