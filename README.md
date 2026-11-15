# Audio Stem Studio (Demucs AI)

Zamonaviy va professional audio stem separator veb-ilovasi. Meta'ning Demucs AI modeli yordamida har qanday qo'shiqni vokal (vocals), baraban (drums), bas (bass) va boshqa asboblar (instruments) ga sifatli ajratadi.

## Imkoniyatlar

- 🎛️ **AI Stem Separation:** 4 ta stem (Vocals, Drums, Bass, Other) yoki 2 ta stem (Vocals, Instrumental) rejimida ajratish.
- 🎚️ **Interaktiv Mikser Deck:** Har bir trek uchun alohida ovoz balandligi (volume), mute, solo sozlamalari.
- 📊 **Real Audio To'lqinlar (Waveforms):** Har bir stem uchun vizual to'lqin shakllari va skrabber (scrubber).
- 🔄 **Sinxron Ijro:** Barcha stemlar birga boshlanadi; ijro davomida har bir stem asosiy trekka nisbatan avtomatik tekislanadi (drift correction).
- 💎 **Lossless stemlar:** Stemlar ichkarida FLAC formatida saqlanadi; brauzerda tinglash uchun yengil MP3 preview ishlatiladi. WAV eksport haqiqiy lossless manbadan olinadi.
- 📦 **ZIP Eksport:** Barcha stemlarni bir marta bosish bilan ZIP arxiv shaklida yuklab olish (MP3 yoki WAV).
- 📱 **Sidebar Navigatsiya:** 
  - **STEM:** Asosiy audio stem ajratish va studiya dekki.
  - **PLAYLIST** va **EDIT:** hozircha faqat joy egallovchi bo'limlar — tez kunda qo'shiladi.

## O'rnatish va Ishga tushirish

### 1. Talablar
- Python 3.10 yoki undan yangi
- FFmpeg: tizimdagi `ffmpeg`/`ffprobe`, virtual muhitdagi binary yoki
  `imageio-ffmpeg` paketidagi binary. `ffprobe` majburiy emas — u bo'lmasa audio
  tekshiruvi `ffmpeg` orqali bajariladi.

### 2. O'rnatish
```bash
# Virtual muhit yaratish
python -m venv .venv

# Virtual muhitni faollashtirish
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Kutubxonalarni o'rnatish
pip install -r requirements.txt
```

### 3. Ishga tushirish
Windows uchun tayyor `start.bat` faylini ishga tushirish yoki:
```bash
python run.py
```

Brauzerda oching:
- Local: http://localhost:8000 yoki http://127.0.0.1:8000

### Ishlash konfiguratsiyasi

Server bitta GPU worker orqali separation tasklarini navbat bilan bajaradi.
Navbatdagi maksimal task soni va saqlash muddatini environment variable orqali
sozlash mumkin:

```bash
# Default qiymatlar
MAX_QUEUED_JOBS=2            # navbatdagi maksimal task soni (0 = cheklanmagan)
TASK_RETENTION_HOURS=24      # tugagan/bekor qilingan tasklar shuncha vaqtdan keyin o'chiriladi
CLEANUP_INTERVAL_MINUTES=60  # tozalash har shuncha daqiqada ishlaydi
MAX_JOB_ATTEMPTS=2           # server qayta ishga tushganda task necha marta qayta urinadi
MAX_CONCURRENT_EXPORTS=2     # bir vaqtda ishlaydigan FFmpeg eksportlari soni
STEM_DATA_DIR=data/tasks     # task fayllari papkasi
```

Upload qilingan audio haqiqiy audio stream bo'lishi, 75 MB dan oshmasligi va
10 soniyadan 10 daqiqagacha davom etishi kerak. Modellar process ichida cache
qilinadi; shu sababli keyingi tasklar modelni qayta yuklamaydi. Export FFmpeg
ishlari alohida thread'da bajariladi va asosiy API loop'ini bloklamaydi.
Processing ekranidagi `Cancel Separation` tugmasi navbatdagi taskni darhol,
ishlayotgan taskni esa keyingi model bo'lagi (chunk) tugashi bilan bekor qiladi.
Bekor qilingan taskning yuklangan fayli va qisman output fayllari darhol
o'chiriladi.

Server ishlash vaqtida qulab tushsa (masalan, xotira yetmasa), task qayta
ishga tushganda yana navbatga qo'yiladi, lekin `MAX_JOB_ATTEMPTS` marta
urinishdan keyin xato sifatida belgilanadi — cheksiz qayta urinish bo'lmaydi.

### Testlar

```bash
pip install -r requirements-dev.txt
python -m pytest
```

### Google Colab (T4 GPU)

`stem_separator_colab.ipynb` notebookidan foydalaning. Colab Runtime sozlamasida
`T4 GPU` tanlangan bo'lishi kerak. Notebook CUDA-enabled PyTorch o'rnatadi va
`torch.cuda.is_available()` hamda GPU nomini tekshiradi; tekshiruv o'tmasa server
ishga tushmaydi.

### Modal.com (serverless T4 GPU)

Modal CLI orqali autentifikatsiya qiling:

```bash
python -m pip install modal
python -m modal setup
```

Deployment:

```bash
python -m modal deploy modal_app.py
```

`modal_app.py` T4 GPU'dan foydalanadi, bitta container bilan ishlaydi va
60 soniya request bo'lmagach scale-to-zero qiladi. Model cache'i va task
fayllari `stem-separator-storage` Modal Volume'ida saqlanadi. Birinchi request
modelni yuklashi sababli odatdagidan uzoqroq davom etadi; keyingi requestlarda
cache ishlatiladi.
