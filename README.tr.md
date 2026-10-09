# RAN Power Optimizer

[English](README.md) | **Türkçe**

Mobil şebekeler (LTE/5G makro katman) için yapay zekâ destekli enerji tasarrufu.
Bir genetik algoritma, her sektörün **iletim gücünü, elektriksel eğimini (tilt) ve uyku
modunu** ayarlar. Amaç, kapsama ve kullanıcı hızı belirlenen sınırlar içinde kalırken ağın
daha az enerji harcamasıdır.

> Tüm veriler **sentetiktir**. Hiçbir operatör verisi, aracı ya da ölçümü kullanılmamıştır.

![Kapsama karşılaştırması](results/coverage_comparison.png)
![Optimizasyon özeti](results/optimization_summary.png)

## Ne yapar?

1. **Ağı kurar:** 3 sektörlü makro baz istasyonlarından oluşan altıgen ızgara, ilişkili
   gölgeleme ve trafik yoğunluk noktaları (hotspot).
2. **Bir yapılandırmayı değerlendirir:** Her nokta için RSRP, SINR, hizmet veren hücre,
   kullanıcı hızı ve toplam giriş gücü.
3. **Optimize eder:** Genetik algoritma; kapsama kaybı, hücre kenarı hızı kaybı ve ortalama
   hız kaybı sınırları altında ağ gücünü en aza indirir. Ardından hücre başına açgözlü
   (greedy) güç iyileştirmesi yapılır.
4. **Raporlar:** Önce/sonra haritaları, yakınsama grafiği, hücre bazlı CSV, KPI JSON'u ve
   Streamlit paneli.

## Hızlı başlangıç

```bash
pip install -r requirements.txt
python main.py                  # 7 baz istasyonu / 21 hücre, çıktılar results/ içinde
python main.py --rings 2        # 19 baz istasyonu / 57 hücre
streamlit run app.py            # etkileşimli panel
python -m pytest                # testler
```

Seçenekler: `--max-cov-drop`, `--max-edge-drop`, `--max-mean-drop`, `--pop`, `--gens`,
`--seed`, `--hotspots`.

## Model

| Bölüm | Yaklaşım |
|---|---|
| Yol kaybı | Basitleştirilmiş 3GPP TR 38.901 UMa, LOS olasılığıyla ağırlıklı, 2.1 GHz, 25 m direk |
| Gölgeleme | Gauss rastgele alanı, 5 dB standart sapma, 80 m korelasyon mesafesi |
| Anten | 3GPP sektör deseni (yatayda 65 derece) ve elektriksel eğim etrafında düşey desen |
| RSRP | Kaynak elemanı (RE) başına güç (1200 alt taşıyıcı) + anten kazancı - yol kaybı |
| SINR | Hizmet veren RSRP / (yük ağırlıklı komşu girişimi + termal gürültü) |
| Hız | Kesilmiş Shannon; hücredeki kullanıcılar bant genişliğini paylaşır |
| Enerji | EARTH tarzı doğrusal model: sabit güç + eğim x RF çıkış gücü, uyku modunda düşük güç |
| Optimizasyon | Genetik algoritma (turnuva seçimi, tekdüze çaprazlama, elitizm) + açgözlü iyileştirme |

Kapsama, RSRP >= -110 dBm ve SINR >= -3 dB anlamına gelir.

## Çıktılar

- `results/coverage_comparison.png`: RSRP, SINR ve hız haritaları, başlangıç ve optimize
- `results/optimization_summary.png`: yakınsama, hücre başına güç, KPI dengesi
- `results/cell_report.csv`: hücre bazlı ayarlar, kullanıcılar ve güç (önce/sonra)
- `results/summary.json`: ağ düzeyinde KPI'lar

## Sınırlamalar

- Statik görüntü: hareket, el değiştirme (handover) ve zamana bağlı trafik yok.
- Basitleştirilmiş yayılım ve örnek değerlerle kurulmuş enerji modeli.
- Kenardaki hücrelerin komşusu daha az olduğu için kenar etkileri vardır.
- Sonuçlar bu simülasyon senaryosuna aittir, gerçek bir şebekeyi temsil etmez.

## Geliştirme fikirleri

- Gece/gündüz trafik profilleri için pekiştirmeli öğrenme
- Çok amaçlı optimizasyon (enerji ve hizmet kalitesi için Pareto cephesi)
- Taktik ya da afet senaryolarında hareketli haberleşme düğümlerinin konumlandırılması

## Kaynaklar

- 3GPP TR 38.901, Study on channel model for frequencies from 0.5 to 100 GHz
- G. Auer ve ark., "How much energy is needed to run a wireless network?", IEEE Wireless Communications, 2011
