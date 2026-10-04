# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Tuấn Cường  
**Mã học viên:** 2A202602717  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8718 | 0.9013 | +0.0295 |
| Answer Relevancy | 0.8958 | 0.9120 | +0.0162 |
| Context Precision | 0.7875 | 0.8650 | +0.0775 |
| Context Recall | 0.8112 | 0.8502 | +0.0390 |

> [!NOTE]
> Tất cả 4 chỉ số cốt lõi của Production RAG đều vượt ngưỡng mục tiêu $\ge 0.75 - 0.85$. Trong đó Context Precision đạt 0.8650 (tăng +0.0775) và Context Recall đạt 0.8502 nhờ việc kết hợp Hierarchical Chunking (Small-to-Big Retrieval), Hybrid Search (BM25 + BGE-M3) và Cross-Encoder Reranking.

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Trích xuất được điều khoản hoàn ứng trong `tam_ung.md` (nêu hạn 15 ngày và phí chậm hoàn ứng 2%/tháng), nhưng thiếu phép tính số ngày quá hạn (20 - 15 = 5 ngày) và số tiền phạt cụ thể (~50.000 VNĐ).
- **Worst metric:** Context Recall (0.5053)
- **Error Tree:** Output sai đáp số cụ thể → Context đúng một phần (chứa công thức nhưng thiếu đáp số) → Query OK (dữ liệu số rõ ràng) → Fix tại Generation/Agent
- **Root cause:** Tài liệu gốc trong doanh nghiệp chỉ quy định điều khoản và công thức lãi suất, không ghi sẵn số tiền phạt cho trường hợp cụ thể 15 triệu và 20 ngày. Hệ thống RAG đơn thuần chỉ làm nhiệm vụ trích xuất (retrieval) chứ không có tầng suy luận logic/toán học (Multi-step Reasoning & Arithmetic Tool).
- **Suggested fix & Trả lời 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng về mặt điều khoản pháp lý nhưng chưa hoàn chỉnh vì thiếu con số tiền phạt cụ thể mà người dùng hỏi.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa điều kiện hạn 15 ngày và lãi suất 2%/tháng, nhưng không có kết quả phép tính 50.000 VNĐ.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không cần, câu hỏi đã cung cấp đủ dữ kiện thực tế.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Bổ sung module ReAct Agent / Tool Calling (Code Interpreter / Calculator) ở tầng LLM Generator để thực thi phép tính số học sau khi trích xuất được công thức.

---

### #2
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Trả lời được thẩm quyền phê duyệt của Giám đốc phòng ban (Director) cho đơn hàng 5-50 triệu, nhưng bỏ sót điều kiện đính kèm 3 báo giá cạnh tranh và xác nhận cấu hình kỹ thuật từ phòng CNTT.
- **Worst metric:** Context Recall (0.6154)
- **Error Tree:** Output thiếu 2 điều kiện ràng buộc → Context chỉ lấy được đoạn phân quyền tài chính, thiếu đoạn quy trình kỹ thuật CNTT → Query đa điều kiện → Fix tại Retrieval
- **Root cause:** Trong tài liệu `mua_sam.md`, quy định về thẩm quyền tài chính (theo mức giá) và quy trình kỹ thuật CNTT (xác nhận cấu hình, 3 báo giá) nằm ở hai mục khác nhau. Do child chunk có kích thước nhỏ, chỉ một trong hai khía cạnh được đưa lên đầu.
- **Suggested fix & Trả lời 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng một phần (chính xác về thẩm quyền Director), thiếu các thủ tục bắt buộc từ phòng CNTT.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Đoạn trích dẫn rank 1 chỉ chứa quy chế hạn mức tài chính, đoạn trích dẫn về CNTT nằm ở rank thấp hơn hoặc bị cắt tỉa.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi chứa 2 vế điều kiện ("ai phê duyệt" và "cần gì từ phòng CNTT").
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 1 (Chunking) cần mở rộng kích thước Parent chunk để bao trọn cả quy trình mua sắm CNTT, đồng thời Module 2 (Search) cần áp dụng Query Decomposition để phân rã truy vấn thành 2 câu hỏi con.

---

### #3
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Đoạn văn trích dẫn ở rank 1 tập trung vào thủ tục thanh toán chi phí đào tạo và hạn mức xét duyệt khóa học, trong khi điều khoản ràng buộc về cam kết thời gian làm việc (1 năm) và mức bồi hoàn 100% nằm ở rank 2.
- **Worst metric:** Context Precision (0.6500)
- **Error Tree:** Output trích dẫn chưa tối ưu rank 1 → Context đúng bị xếp sau đoạn văn chung → Query OK → Fix tại Reranker
- **Root cause:** Bi-Encoder Dense Search tính điểm tương quan cao cho các đoạn chứa nhiều từ khóa "tài trợ khóa học", "chi phí đào tạo 25 triệu" hơn là mệnh đề điều kiện "nghỉ việc sau 8 tháng" (từ ngữ mang tính hoàn cảnh).
- **Suggested fix & Trả lời 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Chưa trực diện, do ngữ cảnh ở rank 1 nói về thủ tục cấp kinh phí thay vì nghĩa vụ bồi hoàn.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đáp án đầy đủ trong Top-3 contexts, nhưng bị tụt xuống rank 2.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rõ ràng, phản ánh đúng bài toán nhân sự thực tế.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 3 (Cross-Encoder Reranking) cần được fine-tune hoặc tinh chỉnh prompt/scoring để phạt nặng các đoạn văn chỉ trùng lặp từ khóa bề mặt mà không giải quyết điều kiện logic của câu hỏi.

---

### #4
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Trả lời nêu chung chung bảng thẩm quyền hoặc nhầm lẫn sang cấp Giám đốc phòng ban (Director - áp dụng cho mức 5-50 triệu).
- **Worst metric:** Context Precision (0.6500)
- **Error Tree:** Output chọn nhầm cấp hạn mức liền kề → Context bảng hạn mức bị cắt vỡ dòng → Query OK → Fix tại Structure Chunking
- **Root cause:** Bảng phân cấp hạn mức mua sắm trong markdown bị chia cắt ngang giữa các dòng khi áp dụng fixed-size splitting hoặc recursive chunking, khiến mối liên kết giữa mức "trên 50 triệu" và chức danh "Tổng Giám đốc (CEO)" bị suy giảm.
- **Suggested fix & Trả lời 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Dễ trả lời sai do nhầm lẫn giữa mốc "đến 50 triệu" (Director) và "trên 50 triệu" (CEO).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa trong tài liệu nhưng bảng bị phân mảnh qua nhiều chunk.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi ngắn gọn, rất chuẩn mực.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 1 (Chunking) cần áp dụng **Structure-aware Chunking** chuyên biệt cho bảng biểu (Table-preserving Chunking), đảm bảo toàn bộ bảng hạn mức thẩm quyền luôn nằm trọn vẹn trong một chunk duy nhất.

---

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Chỉ truy xuất được quy chế nghỉ phép v2024 (`nghi_phep_nam_v2024.md`) hoặc bảng lương (`bang_luong_2024.md`), bỏ sót văn bản còn lại trong Top-3 kết quả trả về.
- **Worst metric:** Context Recall (0.4800)
- **Error Tree:** Output chỉ trả lời được 1 vế (phép năm hoặc lương) → Context bị thiếu 1 nguồn tài liệu hoàn chỉnh (Multi-document synthesis failure) → Query đa miền dữ liệu → Fix tại Query Decomposition
- **Root cause:** Đây là câu hỏi tổng hợp liên tài liệu (Multi-document Question). Một nguồn nằm ở quy chế nhân sự nghỉ phép (`nghi_phep_nam_v2024.md`), nguồn kia nằm ở quy chế lương thưởng (`bang_luong_2024.md`). Khi tính toán độ tương đồng cho cả câu hỏi dài, hệ thống có xu hướng thiên lệch về tài liệu có độ trùng lặp từ khóa cao hơn và đẩy tài liệu còn lại ra khỏi Top-3.
- **Suggested fix & Trả lời 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Chỉ đúng một nửa (chỉ trả lời được số ngày phép hoặc mức lương, không trả lời được cả hai).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Không chứa đủ: Top-3 contexts chỉ chứa 1 trong 2 file tài liệu.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi ghép 2 chủ đề khác biệt vào cùng một câu. Trong thực tế cần tách câu hỏi thành các truy vấn đơn.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Bổ sung module **Query Decomposition / Sub-question Generator**: Tự động bóc tách câu hỏi thành Sub-query 1 ("Nhân viên Senior 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm?") và Sub-query 2 ("Khung lương của nhân viên Senior là bao nhiêu?"), thực hiện retrieval độc lập rồi tổng hợp kết quả (Merge Contexts).

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
`Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?`

### Error Tree Walkthrough:
1. **Output đúng?**  
   → **Không hoàn toàn**: Mô hình chỉ trả lời được vế ngày phép (18 ngày) nhưng bỏ sót hoặc trả lời mơ hồ về khung lương (20 - 35 triệu VNĐ).
2. **Context đúng?**  
   → **Không đủ**: Context Recall bị tụt xuống mức 0.4800 do Top-3 chỉ gom được các chunk thuộc file `nghi_phep_nam_v2024.md`, trong khi các chunk của file `bang_luong_2024.md` bị xếp ở rank 6-8 và bị bộ lọc `top_k=3` cắt bỏ.
3. **Query rewrite OK?**  
   → **Chưa tối ưu**: Câu hỏi nguyên bản là câu hỏi phức hợp đa mục tiêu (Multi-intent Query) kết hợp giữa chính sách nghỉ phép và chính sách lương bổng.
4. **Fix ở bước:**  
   → Bổ sung tầng **Query Decomposition** ở tiền xử lý và tăng `top_k` của tầng Retrieval trước khi đưa vào Cross-Encoder Reranker.

### Nếu có thêm 1 giờ, sẽ optimize:
1. **Triển khai Sub-question Query Engine (Agentic Routing):**
   - Sử dụng LLM phân tích câu hỏi người dùng thành 2 truy vấn độc lập:
     - `Query A`: "Chính sách phép năm nhân viên thâm niên 9 năm v2024" → Tìm kiếm trong danh mục HR/Leave.
     - `Query B`: "Bảng lương vị trí Senior P3 P4 năm 2024" → Tìm kiếm trong danh mục Finance/Salary.
   - Hợp nhất (concatenate) Top-2 contexts từ mỗi truy vấn con để đảm bảo LLM có đầy đủ 100% ngữ cảnh của cả hai tài liệu.
2. **Gắn Metadata Filter tự động theo năm ban hành:**
   - Thêm quy tắc lọc cứng `metadata["version"] == "v2024"` cho các truy vấn về chính sách hiện hành để loại bỏ hoàn toàn nguy cơ lấy nhầm văn bản v2023 cũ.
