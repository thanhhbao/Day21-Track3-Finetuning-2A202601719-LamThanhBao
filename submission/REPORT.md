# Lab 21 — Evaluation Report

**Họ tên**: Lâm Thành Bảo **MSSV**: 2A202601719 **Ngày**: 2026-08-21
**Tier**: `T4` **Base model**: `unsloth/Qwen3.5-4B` **GPU thực tế**: `Colab Free T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

|                    |                                                            |
| ------------------ | ---------------------------------------------------------- |
| Dataset            | 250 ticket CSKH → JSON triage (corpus mặc định, không đổi) |
| Train / val        | 225 / 25 (seed 42)                                         |
| `max_length`       | 1024 — p95 đo được là 98 _(results/token_stats.json)_      |
| `MASK_MODE`        | `assistant-only`                                           |
| Epochs / max_steps | 2 / 30                                                     |

**Template có giữ khối `<think>` không?** `có` — _(results/template_check.json, verdict: "reasoning preserved — safe to train on traces")_

**Vì sao `max_length=1024` dù p95 chỉ gợi ý 256?** Giữ nguyên mặc định của tier `T4` thay vì hạ theo p95: 1024 là trần an toàn của tier, không phải mục tiêu cần khớp chính xác — hạ xuống 256 chỉ tiết kiệm được VRAM/tốc độ không đáng kể ở batch nhỏ (effective batch < 32 theo rule của lab), trong khi vẫn còn dư địa nếu có vài ticket dài bất thường lọt vào p99–max (100–101 token) hoặc nếu đổi sang corpus khác dài hơn. Không hạ xuống vì lợi ích không đủ bù rủi ro cắt mất dữ liệu.

---

## 2. Mask proof (NB1)

|                              |          |
| ---------------------------- | -------- |
| `supervised_fraction`        | `0.4149` |
| Câu trả lời nằm trong loss   | `true`   |
| Câu hỏi KHÔNG nằm trong loss | `true`   |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run                         | target | regression | format | latency (ms) |
| --------------------------- | ------ | ---------- | ------ | ------------ |
| (a) base + naive prompt     | 0.000  | 0.758      | 0.000  | 3282.6       |
| (b) base + optimized prompt | 0.765  | 0.758      | 1.000  | 1022.0       |
| (c) LoRA fine-tune          | 0.955  | 0.756      | 1.000  | 1491.9       |

**(b) có thật sự mạnh hơn (a) không?** `có` — target nhảy từ 0.000 lên 0.765 và format từ 0.000 lên 1.000 chỉ nhờ đổi prompt, chứng tỏ base model hoàn toàn có khả năng làm đúng tác vụ nếu được hướng dẫn tử tế; đây là mốc thật để fine-tune phải vượt qua, không phải một baseline yếu dựng lên cho dễ thắng.
`OPTIMIZED_PROMPT` không bị sửa (`make verify` xác nhận `baseline (b) prompt unmodified`), nên (b) là mốc nguyên bản của lab, không bị làm yếu đi.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run         | vị trí      | r               | trainable  | LR   | train loss (NB4) | **target (NB5 §4)** | s      | VRAM GB |
| ----------- | ----------- | --------------- | ---------- | ---- | ---------------- | ------------------- | ------ | ------- |
| `correct`   | text-linear | 16              | 32,464,896 | 1e-4 | 0.6771           | **0.955**           | 950.0  | 12.01   |
| `attn_only` | q,v         | _283 (matched)_ | 32,456,704 | 1e-4 | 0.5368           | **0.970**           | 815.5  | 12.02   |
| `wrong_lr`  | text-linear | 16              | 32,464,896 | 1e-5 | 1.5704           | **0.000**           | 957.3  | 12.01   |
| `qlora`     | text-linear | 16              | 32,464,896 | 1e-4 | 0.7058           | **0.940**           | 1041.3 | 7.09    |

> Thứ tự theo cột target ở đây trùng với thứ tự theo train loss (`attn_only` tốt nhất ở cả hai, rồi `correct`, `qlora`, `wrong_lr` tệ nhất) — xem giải thích ở 4.1, đây không phải lý do để tin train loss thay cho target.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1** — `attn_only` có cùng số tham số huấn luyện với `correct` (32.456.704 vs 32.464.896, lệch <0,03% — đạt ngưỡng công bằng <5% mà `make verify` kiểm tra). Trên tập target nó **thắng**: 0.970 so với 0.955 của `correct`. Thứ tự đó **giống** thứ tự theo train loss (`attn_only` có loss thấp nhất 0.5368, `correct` 0.6771). Điều bất ngờ là deck gọi `attn_only` là "Mistake #1" (đặt sai vị trí adapter), nhưng ở đúng ngân sách tham số và đúng 30 step này, việc chỉ gắn vào `q,v` với rank cao hơn lại nhỉnh hơn việc gắn vào toàn bộ lớp tuyến tính với rank thấp. Điều này gợi ý _rank_ (số tham số huấn luyện) không phải là yếu tố quyết định duy nhất, và ở quy mô nhỏ (30 step, tập eval 50 mẫu, bài toán triage đơn giản 4 trường) thì vị trí `attn-only` với rank được bù đắp đủ lớn vẫn đủ biểu đạt — không nên vội kết luận `attn_only` luôn thua chỉ vì đó là "cấu hình sai" theo lý thuyết; cần thêm seed/run lặp lại để chắc chắn đây không phải nhiễu của tập eval 50 mẫu.

**4.2** — `wrong_lr` chỉ khác đúng learning rate (1e-5 thay vì 1e-4, tức LR ở thang full-fine-tune áp cho LoRA). Đường loss của nó dừng ở 1.5704 — cao hơn hẳn `correct` (0.6771) — cho thấy mô hình gần như không học được gì trong 30 step với LR quá nhỏ so với LoRA. Hậu quả trên downstream còn nghiêm trọng hơn con số loss thể hiện: target và format đều rơi về **0.000** — output JSON hỏng hoàn toàn (`format=0`), không chỉ là "học chậm". Nếu chỉ nhìn train loss mà không biết LR, người ta dễ kết luận nhầm là mô hình "underfit nhẹ, cần train thêm step" — trong khi thực tế là cấu hình sai khiến output hoàn toàn không dùng được; chỉ số loss một mình không lộ ra điều đó, phải đo trên tác vụ thật (`format`, `target`) mới thấy.

**4.3** — `qlora` tiết kiệm VRAM đỉnh từ 12.01 GB xuống 7.09 GB (giảm ~41%, ~4.92 GB), nhưng trả giá bằng: target thấp hơn một chút (0.940 vs 0.955 của `correct`, -0.015), thời gian train lâu hơn (1041.3s vs 950.0s, +9.6%), và đặc biệt latency inference cao hơn rõ rệt (1871.1ms vs 1491.9ms, +25.4%) do phải dequantize 4-bit khi generate. Ở tier T4 16GB, `correct` (16-bit) đã chạy thoải mái trong 12.01 GB, tức không có áp lực VRAM thật sự buộc phải dùng QLoRA. Số đo của tôi **ủng hộ** khuyến nghị "không dùng QLoRA cho dòng model này" ở quy mô lab này: cái giá phải trả (chậm hơn, kém chính xác hơn một chút) không đáng so với lượng VRAM tiết kiệm được, khi VRAM sẵn có đã đủ dùng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.190` · `regression Δ = -0.002` · `valid_trace_rate = 0.00`

Diễn giải: Bản fine-tune vượt baseline (b) đã prompt tối ưu ở nhóm target (+0.190, từ 0.765 lên 0.955) trong khi gần như không đánh đổi năng lực tổng quát — nhóm regression chỉ giảm 0.002 (0.7578 → 0.7556 trên 15 câu hỏi kiến thức phổ thông), nằm sâu trong ngưỡng chấp nhận được. `format` đạt tuyệt đối 1.000 (không khác biệt so với (b)), nên phần thắng chủ yếu đến từ việc phân loại đúng nội dung (intent/urgency/product/sentiment) chứ không phải từ việc học cách xuất JSON hợp lệ — (b) đã làm được điều đó rồi nhờ prompt tốt. `valid_trace_rate = 0.00` không phải dấu hiệu xấu ở đây: corpus mặc định không có khối `<think>` thật nào trong câu trả lời (chỉ là `<think></think>` rỗng do template tự đóng), nên chỉ số này không có gì để đo — nó chỉ có ý nghĩa khi chạy thử nghiệm bonus B3 với corpus có reasoning trace thật.

---

## 6. Định tính — bắt buộc có cả ca THUA

| #   | Ticket (rút gọn)                                                          | Nhãn đúng                                        | (b) prompt             | (c) fine-tune                            | Nhận xét                   |
| --- | ------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------- | ---------------------------------------- | -------------------------- |
| 1   | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền.    | urgency=`thap`                                   | _(không lưu per-item)_ | urgency=`trung_binh` (sai, còn lại đúng) | ❌ **FT thua** so với nhãn |
| 2   | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện.    | urgency=`thap`                                   | _(không lưu per-item)_ | urgency=`trung_binh` (sai, còn lại đúng) | ❌ **FT thua** so với nhãn |
| 3   | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện.     | urgency=`thap`                                   | _(không lưu per-item)_ | urgency=`trung_binh` (sai, còn lại đúng) | ❌ **FT thua** so với nhãn |
| 4   | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô... | intent=`van_chuyen`, urgency=`thap`, ...         | _(không lưu per-item)_ | khớp 4/4 trường                          | ✅ FT thắng                |
| 5   | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n... | intent=`san_pham_loi`, urgency=`trung_binh`, ... | _(không lưu per-item)_ | khớp 4/4 trường                          | ✅ FT thắng                |

> **Lưu ý minh bạch**: `notebooks/05_evaluate_and_verdict.py` chỉ ghi `results/qualitative.json` với điểm/dự đoán của **fine-tune**, không lưu lại dự đoán từng câu của baseline (b) — nên cột "(b) prompt" ở trên không điền được bằng dữ liệu đã chạy. 5 ví dụ trên được chọn từ 50 mẫu target theo `ft_score` (3 điểm thấp nhất = 0.75/1.0, 2 điểm cao nhất = 1.0/1.0), so với **nhãn đúng** — đúng với cách NB5 tự định nghĩa "3 ca tệ nhất / 3 ca tốt nhất".

**Có mẫu chung nào ở các ca FT thua không?** Có — cả 3 ca thua đều sai **cùng một trường**: `urgency`, và cùng một kiểu sai: nhãn đúng là `thap` (thấp) nhưng model luôn đoán `trung_binh` (trung bình). Cả 3 ticket đều diễn đạt mức độ khẩn cấp một cách ngầm định, không dùng từ khóa rõ ràng như "gấp" hay "khẩn" — có vẻ model có thiên hướng đoán "trung bình" làm mặc định khi không thấy tín hiệu khẩn cấp rõ ràng, thay vì suy luận ra mức "thấp".

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Bản fine-tune này đáng để deploy cho bài toán triage CSKH tiếng Việt: nó vượt baseline (b) — vốn đã là base model với prompt tối ưu, không phải một baseline yếu — ở nhóm target (+0.190) mà gần như không đánh đổi năng lực tổng quát (regression chỉ giảm 0.002, nằm trong ngưỡng chấp nhận). Cổng hồi quy 4 nhóm PASSED, và phần thắng đến từ đúng chỗ đáng tin: phân loại nội dung chính xác hơn, không phải từ việc "gian lận" bằng cách chỉ học cách xuất JSON đúng định dạng (format đã đạt tuyệt đối ở cả (b) và (c)). Về đòn bẩy thật sự trong lab: dữ liệu của tôi cho thấy bức tranh phức tạp hơn một "đòn bẩy duy nhất" — learning rate là đòn bẩy rõ ràng nhất và nguy hiểm nhất (`wrong_lr` sập hoàn toàn về target=0, format=0 chỉ vì lệch LR một bậc), trong khi vị trí adapter (`attn_only` vs `text-linear`) ở ngân sách tham số bằng nhau lại cho kết quả ngược với kỳ vọng lý thuyết (attn_only thắng nhẹ) — điều này nhắc tôi rằng ở quy mô lab nhỏ (30 step, 50 mẫu eval), kết luận về vị trí cần thêm run lặp lại mới đủ tin cậy, còn kết luận về LR thì rất chắc chắn chỉ với một lần đo vì mức độ sập là quá rõ ràng để là nhiễu ngẫu nhiên. Chất lượng dữ liệu (mask đúng, corpus sạch) là điều kiện cần nhưng không phải là biến số tôi thử nghiệm trực tiếp trong lab này.

**Ba điều tôi học được** (cụ thể, không generic):

1. Đo trên chỉ số thay thế (train loss) và đo trên tác vụ thật (target/format) có thể **đồng thuận** hoặc **mâu thuẫn** tùy run — trong lab này chúng đồng thuận cho cả 4 run, nhưng `wrong_lr` cho thấy rõ nhất: loss tăng gấp đôi không nói lên được rằng output đã sập hoàn toàn về JSON rỗng/hỏng (format=0), phải đo downstream mới thấy mức độ nghiêm trọng thật.
2. "Cấu hình sai theo lý thuyết" (deck gọi `attn_only` là Mistake #1) không tự động thua trong mọi điều kiện thực nghiệm — khi ngân sách tham số được bù đắp công bằng (`matched_rank`), nó thắng nhẹ ở lab cụ thể này với 30 step. Bài học không phải "attn_only tốt hơn" mà là: kết luận về vị trí adapter cần nhiều run/seed hơn một lần đo để tách tín hiệu khỏi nhiễu của tập eval 50 mẫu.
3. `results/qualitative.json` do NB5 sinh ra chỉ lưu dự đoán của fine-tune, không lưu dự đoán từng câu của baseline (b) — nghĩa là bảng so sánh định tính trực tiếp (b) vs (c) theo đúng mẫu report không thể điền đầy đủ chỉ từ artefact có sẵn; lần sau cần chủ động lưu thêm `preds_b` song song với `preds_ft` trong NB5 nếu muốn báo cáo so sánh từng ví dụ thật sự công bằng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

- Sửa NB5 để lưu thêm dự đoán per-item của baseline (b), rồi dựng lại bảng định tính so sánh trực tiếp (b) vs (c) từng câu thay vì chỉ so với nhãn.
- Chạy lại `attn_only` với 2-3 seed khác để kiểm tra kết quả thắng `correct` ở mục 4.1 có ổn định hay chỉ là nhiễu của tập eval 50 mẫu.
- Thử làm rõ thiên hướng đoán `urgency=trung_binh` khi thiếu tín hiệu rõ ràng — có thể do phân bố nhãn `urgency` trong 250 mẫu train bị lệch, cần kiểm tra và cân bằng lại.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
