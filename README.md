# rag-lab

**Framework kullanmadan**, her adımı gözlenebilir bir RAG (Retrieval-Augmented Generation)
demosu. Amaç öğrenmek: yükle → parçala → embed → ara → Claude ile cevapla → ölç.

```
İNDEKSLEME   sample_docs/ ─load─► docs.jsonl ─chunk─► chunks.jsonl ─embed─► index.npz
SORGU        soru ─embed─► search (kosinüs, top-k) ─► answer (prompt + Claude) ─► cevap + [kaynak]
ÖLÇÜM        eval.jsonl ─► evaluate.py ─► hit@k + cevap isabeti
```

Varsayılan korpus `sample_docs/`: hayali bir kahve atölyesinin el kitabı (Markdown + HTML).
Kendi dokümanlarınla denemek için `RAG_SOURCE=/yol/klasor` ver. Desteklenen türler:
`.md`, `.html`, `.pdf`, `.docx`.

## Kurulum

```bash
uv sync
brew install ollama && ollama serve     # ayrı terminalde açık kalsın
ollama pull bge-m3                      # ~1.2 GB, çok dilli embedding modeli
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # git'e girmez
```

## Kullanım

```bash
uv run python index.py                              # hepsi: load → chunk → embed
uv run python index.py --step chunk                 # tek adım (öncekinin çıktısı data/'da olmalı)

uv run --env-file .env python ask.py "soru"         # cevap + kaynaklar
uv run --env-file .env python ask.py                # etkileşimli mod (REPL)
uv run python ask.py --retrieve-only "soru"         # Claude'suz: yalnız hangi parçalar geliyor
uv run --env-file .env python ask.py --debug "soru" # parçaların tam metni + Claude'a giden prompt
uv run --env-file .env python ask.py --trace "soru" # her adımı görselleştiren HTML → data/traces/

uv run --env-file .env python evaluate.py           # ölçüm → data/eval_<tarih>.txt
uv run python evaluate.py --retrieve-only -k 2      # yalnız arama, farklı k (örnek korpus küçük: 7 parça)
uv run pytest                                       # birim testleri
```

Örnek sorular:

```bash
uv run --env-file .env python ask.py "Model C kavurma makinesi neden seçilmedi?"
uv run --env-file .env python ask.py "Ada Kahve'nin 2022 cirosu ne kadardı?"   # tuzak: "bulamadım" beklenir
```

Ara çıktılar `data/` altındadır ve hepsi açılıp okunabilir:
- `docs.jsonl`: düz metin
- `load_report.txt`: atlanan, kopya ve hatalı dosyalar
- `chunks.jsonl`: parçalar
- `index.npz`: vektörler

## İz sayfası (`--trace`)

Tek dosyalık, çevrimdışı çalışan bir HTML üretir. Tarayıcıda açılır. Sayfada şunlar var:
- **Büyük resim:** İndeksleme ve sorgu aşamaları, gerçek sayılarla.
- **Süreler:** Her adımın toplam süreye oranı.
- **Embed:** Sorunun 1024 boyutlu vektörü, ısı şeridi olarak.
- **Arama:**
  - Tüm skorların histogramı ve k kesim çizgisi
  - "Kıl payı" farkı: k. parça ile k+1. parça arasındaki skor farkı
  - Klasöre göre skor şeridi
  - PCA anlam haritası: tüm parçalar, soru ★ ve seçilenler
  - Seçilen parçalar ve kesimin hemen altında kalanlar
- **Prompt:** Metnin bileşimi, talimat ve parçaların tam hâli.
- **Claude:** Token sayıları; cevaptaki `[n]` atıflarının üzerine gelince ilgili parça vurgulanır.

## Uçak kazası raporları (`corpora/aci/`)

İkinci korpus: [hayatibis.dev'deki Air Crash Investigation indeksinden](https://hayatibis.dev/writing/air-crash-investigation-index)
seçilmiş 25 resmî kaza raporu (NTSB, BEA, AAIB, TSB, KNKT). Raporlar repoda değil; `fetch.py`
kaynağından indirir, `data/aci/` altına koyar (gitignore'da).

```bash
brew install ocrmypdf                                   # taranmış sayfalar için OCR
uv run python corpora/aci/fetch.py                      # indir → OCR → soru seti → istatistik

# aynı raporlardan iki indeks: OCR'sız ve OCR'lı
RAG_SOURCE=data/aci/pdf RAG_DATA=data/aci/index-raw uv run python index.py
RAG_SOURCE=data/aci/ocr RAG_DATA=data/aci/index-ocr uv run python index.py

# aynı sorularla üç arama: dense (kosinüs), BM25, hibrit (RRF); dosya ve parça seviyesinde
RAG_DATA=data/aci/index-ocr RAG_EVAL=data/aci/eval.jsonl uv run python corpora/aci/compare.py

# yayınlanabilir, önceden hesaplanmış demo → data/aci/demo/index.html + traces/
RAG_DATA=data/aci/index-ocr uv run --env-file .env python corpora/aci/demo.py
```

- `manifest.json`: seçilen raporlar, resmî rapor linkleri ve indeksteki kısa neden özeti.
- `fetch.py`: `ocrmypdf --skip-text` yalnız metni olmayan sayfaları okur; `stats.tsv` rapor başına
  OCR'sız ve OCR'lı karakter sayısını yazar.
- `eval.jsonl` otomatik üretilir (rapor başına adıyla ve nedeniyle iki soru + tuzaklar). Daha sıkı
  ölçüm için elle yazılmış, beklenen sayfa ve ifadeyle bir set (`RAG_EVAL=… compare.py`) kullanılır;
  parça seviyesi, ilk k parçadan birinin doğru rapordan gelip beklenen ifadeyi içermesidir.
- `compare.py` içindeki BM25 ve RRF kütüphanesiz, elle yazılmıştır.

## Öğrenilenler

1. **Veri hazırlığı sessizce bozulur.** HTML tablosunda `#` sütunu `# | Madde | …` satırı
   üretiyor, parçalayıcı bunu H1 sanıyordu. Birim testleri geçiyordu; hatayı ancak gerçek
   çıktıya bakmak gösterdi. Ders: indekslemeden sonra parçaları **gözle** oku.
2. **Anlamsal arama, birebir ifadeyi kaçırabilir.** Cevabı içeren parça, soruda geçen
   anahtar kelimeyi değil eş anlamlısını kullanınca ilk k'ya giremeyebilir. Kelime tabanlı
   arama (BM25) bunu yakalar → **hibrit arama**.
3. **Konu kalabalığı azınlığı boğar.** Aynı konuda çok sayıda doküman varsa, kuralı içeren
   tek kısa satır geride kalır. Kısa ve bağlamsız metin zayıf vektör üretir.
4. **Ölçüt de hata yapar.** Dosya seviyesinde hit@k, doğru bilgiyi başka dosyadan getiren
   başarıları ✗, doğru dosyanın yanlış parçasını ✓ sayar. Ölçüm parça seviyesine inmeli.
5. **Retrieval/generation ayrımı işe yarar.** `--retrieve-only` + `--debug` ile her ıskanın
   hangi katmanda olduğu hızla belli olur.

## Sonraki adımlar

- LlamaIndex ile aynı korpus + aynı `eval.jsonl` üzerinde kıyas
- Hibrit arama (BM25 + vektör)
- Parça seviyesinde ölçüt (beklenen ifade getirilen parçalarda geçiyor mu)
- k (5 / 8 / 12) ve parça boyutu (400 / 800 / 1200 token) taraması
- Reranker (getirilen 30 parçayı ikinci bir modelle yeniden sıralamak)
