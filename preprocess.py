import os
import shutil
import zipfile
from data import kss
import hparams as hp

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def safe_move(src, dst):
    """Windows에서 안전하게 파일/폴더 이동"""
    if os.path.exists(src):
        try:
            shutil.move(src, dst)
        except Exception:
            pass

def unzip_safe(zip_path, out_dir):
    """Windows에서 unzip"""
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)

def main():
    in_dir = hp.data_path              # KSS 폴더
    out_dir = hp.preprocessed_path     # preprocessed/
    meta = hp.meta_name                # transcript.v.1.4.txt
    textgrid_name = hp.textgrid_name   # TextGrid.zip


    for d in ["mel", "alignment", "f0", "energy"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)


    tg_src = os.path.join(".", textgrid_name)
    tg_dst = os.path.join(out_dir, textgrid_name)

    if os.path.isfile(tg_src):
        safe_move(tg_src, tg_dst)

    tg_folder = os.path.join(out_dir, textgrid_name.replace(".zip", ""))

    if not os.path.exists(tg_folder):
        unzip_safe(tg_dst, out_dir)

    if "KSS" in hp.dataset.upper() and "v.1.4" in meta:

        wavs = os.path.join(in_dir, "wavs")
        wavs_bak = os.path.join(in_dir, "wavs_bak")

        if not os.path.exists(wavs_bak):
            os.makedirs(wavs, exist_ok=True)


            for i in range(1, 5):
                src = os.path.join(in_dir, str(i))
                if os.path.exists(src):
                    for f in os.listdir(src):
                        safe_move(os.path.join(src, f), wavs)

            safe_move(wavs, wavs_bak)
            os.makedirs(wavs, exist_ok=True)

    # ------------------------------------------
    # 3) 전처리 시작
    # ------------------------------------------
    train, val = kss.build_from_path(in_dir, out_dir, meta)
    write_metadata(train, val, out_dir)

if __name__ == "__main__":
    main()
