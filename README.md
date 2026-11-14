# Audio Stem Studio (Demucs AI)

Zamonaviy va professional audio stem separator veb-ilovasi. Meta'ning Demucs AI modeli yordamida har qanday qo'shiqni vokal (vocals), baraban (drums), bas (bass) va boshqa asboblar (instruments) ga sifatli ajratadi.

## Imkoniyatlar

- 🎛️ **AI Stem Separation:** 4 ta stem (Vocals, Drums, Bass, Other) yoki 2 ta stem (Vocals, Instrumental) rejimida ajratish.
- 🎚️ **Interaktiv Mikser Deck:** Har bir trek uchun alohida ovoz balandligi (volume), mute, solo sozlamalari.
- 📊 **Real Audio To'lqinlar (Waveforms):** Har bir stem uchun vizual to'lqin shakllari va skrabber (scrubber).
- 🔄 **Sinxron Ijro:** Barcha stemlar bir xil vaqtda sinxron ijro etiladi.
- 📦 **ZIP Eksport:** Barcha stemlarni bir marta bosish bilan ZIP arxiv shaklida yuklab olish.
- 📱 **Sidebar Navigatsiya:** 
  - **PLAYLIST:** Pleylistlar va audio kutubxona bo'limi.
  - **STEM:** Asosiy audio stem ajratish va studiya dekki.
  - **EDIT:** To'lqin shakllari va audioni tahrirlash bo'limi.

## O'rnatish va Ishga tushirish

### 1. Talablar
- Python 3.10 yoki 3.11
- FFmpeg (tizimda yoki virtual muhitda mavjud bo'lishi kerak)

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
