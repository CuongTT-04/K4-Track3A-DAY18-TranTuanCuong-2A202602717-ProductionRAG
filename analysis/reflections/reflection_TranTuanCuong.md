# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Tuấn Cường  
**Mã học viên:** 2A202602717  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code đã triển khai trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Sử dụng Cosine Similarity giữa các câu liên tiếp (thông qua embedding model `all-MiniLM-L6-v2`) với ngưỡng `threshold=0.85`. Khi độ tương đồng giữa hai câu liên tiếp giảm xuống dưới ngưỡng, một điểm ngắt đoạn (split point) được tạo ra. Kỹ thuật này giúp bảo toàn ngữ nghĩa trọn vẹn của luận điểm thay vì cắt mù quáng theo số lượng ký tự như fixed-size chunking. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Cắt tài liệu thành Parent Chunks (kích thước lớn ~1000 ký tự) nhằm bao quát đầy đủ ngữ cảnh cho LLM đọc hiểu, sau đó chia nhỏ thành Child Chunks (~200 ký tự) liên kết qua `parent_id` để tăng độ tập trung và chính xác cho vector search. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF kết hợp điểm thứ hạng từ Lexical Search (BM25Okapi với tách từ tiếng Việt bằng `underthesea`) và Semantic Search (Dense Search bằng `BAAI/bge-m3` đa ngôn ngữ lưu trên Qdrant Vector DB). Hằng số `k=60` giúp cân bằng thứ hạng, đảm bảo vừa bắt chính xác các từ khóa hiếm (mã văn bản, ngày tháng, con số tài chính) vừa nắm bắt được câu hỏi diễn đạt tự nhiên theo ngữ nghĩa. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` và tính toán attention chéo qua tất cả các tầng transformer. Mặc dù độ trễ cao hơn dense retrieval, việc đưa top-20 ứng viên từ M2 qua Cross-Encoder giúp lọc nhiễu triệt để, đẩy đoạn văn mang đáp án thực sự lên top 1-3, tăng vọt chỉ số Context Precision. |
| RAGAS 4 metrics & Diagnostic Tree | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá toàn diện 4 khía cạnh: Faithfulness (độ trung thực của câu trả lời dựa trên context), Answer Relevancy (mức độ bám sát câu hỏi), Context Precision (tỷ lệ context đúng ở vị trí xếp hạng cao), và Context Recall (khả năng bao quát đầy đủ thông tin ground truth). Sử dụng Cây chẩn đoán (Diagnostic Tree) để tự động phân loại nguyên nhân gốc rễ và đề xuất giải pháp cho từng truy vấn bị điểm thấp. |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()` | Khắc phục hiện tượng "mất ngữ cảnh" khi tài liệu bị chia nhỏ bằng cách sinh tiêu đề ngữ cảnh tóm tắt vị trí, chủ đề và thời gian ban hành (năm 2023 vs 2024), ghép vào đầu mỗi chunk trước khi index. Đồng thời trích xuất metadata và sinh các câu hỏi giả định (Hypothetical Questions) để mở rộng không gian tìm kiếm. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. *Lỗi môi trường Python:* Khi cài đặt dependencies trên Python 3.14: `error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"` do các thư viện khoa học dữ liệu (`numpy 1.26.4`, `hdbscan`, `pandas`, `ragas`) chưa phát hành pre-built binary wheels cho Python 3.14.
  2. *Lỗi bảo mật Windows Smart App Control (SAC):* Khi chạy các module AI: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed: c10.dll` do chính sách Code Integrity của Windows 11 chặn DLL của PyTorch khi thực thi trong môi trường sandbox cô lập.
  3. *Lỗi tách từ tiếng Việt:* Thư viện `underthesea.word_tokenize` nối từ ghép bằng dấu gạch dưới (ví dụ `nghị_định`, `bảo_vệ`), khiến tìm kiếm từ khóa không khớp với truy vấn người dùng nhập dạng từ rời thông thường.
  4. *Lỗi OpenAI API key giả lập:* File `.env` cấu hình `OPENAI_API_KEY=sk-...` (placeholder). Khi gọi API OpenAI sinh câu trả lời hoặc chạy RAGAS qua mạng, client liên tục gặp lỗi `AuthenticationError: Error code: 401 - Incorrect API key provided`, dẫn đến nghẽn tiến trình và kéo dài thời gian chờ đợi.

- **Nguyên nhân gốc rễ & Cách debug:**
  1. *Xử lý môi trường:* Tạo lại virtual environment `.venv` chuẩn sử dụng Python 3.11.9 64-bit có sẵn trên hệ thống (`pythoncore-3.11-64`). Tất cả các thư viện `torch`, `sentence-transformers`, `qdrant-client`, `ragas`, `datasets` đều có sẵn binary wheel tương thích hoàn toàn, không yêu cầu biên dịch C++.
  2. *Xử lý DLL Windows:* Chạy script thông qua đường dẫn python trực tiếp của venv (`.\.venv\Scripts\python.exe`) và cấu hình thực thi ngoài sandbox an toàn để cấp quyền đầy đủ cho Windows nạp các DLL toán học tối ưu (`torch_cpu.dll`, `c10.dll`).
  3. *Xử lý tách từ:* Thêm bước chuẩn hóa `token.replace("_", " ")` trong hàm `segment_vietnamese()` để đảm bảo BM25Okapi lập chỉ mục và tra cứu đồng nhất với token văn bản tự nhiên.
  4. *Xử lý API Key & Heuristic Fallback:* Viết hàm kiểm tra tính hợp lệ của API key `_has_valid_api_key()`. Nếu key không hợp lệ hoặc là placeholder, hệ thống chuyển đổi mượt mà sang cơ chế trích xuất heuristic cục bộ và đánh giá IR/semantic similarity theo chuẩn RAGAS, đảm bảo toàn bộ pipeline chạy độc lập và ổn định 100%.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sâu hơn về kiến trúc bất đối xứng (Bi-Encoder vs Cross-Encoder) và sự đánh đổi giữa Latency và Accuracy trong hệ thống Production. Đã bổ sung kiến thức qua việc đo đạc thời gian thực thi: Bi-Encoder (Dense Search) tìm kiếm nhanh qua Vector Index nhưng thiếu tương tác sâu giữa query và doc, trong khi Cross-Encoder tính toán ma trận Attention toàn phần nên phù hợp tối ưu làm tầng re-ranking thứ hai cho tập ứng viên nhỏ.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào dự án:

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy chế Doanh nghiệp (Enterprise Legal Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: Chia nhỏ văn bản dạng recursive character splitting cố định 500 ký tự, sinh vector embedding qua OpenAI text-embedding-3-small và lưu trên ChromaDB, truy vấn thuần túy bằng Cosine Similarity.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Xung đột phiên bản (Version Conflicts):* Khi truy vấn về chính sách thuế, bảo hiểm hoặc an toàn thông tin, hệ thống thường nhầm lẫn giữa thông tư cũ (2022/2023) và nghị định mới cập nhật (2024).
  - *Mất ngữ cảnh (Context Fragmentation):* Các điều khoản pháp lý thường viện dẫn tới các chương, mục nằm ở phần đầu tài liệu. Khi chunk bị cắt rời, thông tin về "Chương nào, Mục nào" bị mất khiến câu trả lời không chính xác.
  - *Bỏ sót điều khoản cụ thể:* Vector search thuần túy thường bỏ qua số hiệu văn bản chính xác (ví dụ "Nghị định 13/2023/NĐ-CP").

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:**
   - Kết hợp **Structure-aware chunking** theo cây phân cấp pháp lý (Chương > Điều > Khoản > Điểm) và **Hierarchical chunking** (Parent chunk lưu toàn bộ Điều luật kèm tiêu đề Chương, Child chunk lưu từng Khoản).
2. **Search retrieval:**
   - Áp dụng **Hybrid Search**: BM25 kết hợp tách từ `underthesea` để bắt chính xác các số hiệu điều luật, ngày hiệu lực và các con số mức phạt; kết hợp Dense Retrieval dùng `BAAI/bge-m3` để bắt ngữ nghĩa tự nhiên của câu hỏi.
   - Hợp nhất kết quả bằng **Reciprocal Rank Fusion (RRF)** với tham số $k=60$.
3. **Reranking:**
   - Tích hợp tầng **Cross-Encoder Reranking** sử dụng `BAAI/bge-reranker-v2-m3` lấy Top 25 ứng viên từ Hybrid Search và xếp hạng lại để chọn ra Top 3-5 ngữ cảnh chính xác nhất trước khi gửi vào LLM.
4. **Evaluation:**
   - Xây dựng bộ test-set gồm 50 câu hỏi nghiệp vụ thực tế có gán nhãn ground-truth.
   - Đo lường tự động định kỳ bằng 4 chỉ số RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall) với ngưỡng mục tiêu $\ge 0.80$.
5. **Enrichment:**
   - Áp dụng **Contextual Prepending**: Tự động chèn metadata bao gồm: Tên văn bản, Số hiệu, Ngày ban hành, Trạng thái hiệu lực (Đang có hiệu lực / Đã hết hiệu lực) vào đầu mỗi chunk.
   - Thêm câu hỏi giả định (Hypothetical Questions) hỗ trợ các câu hỏi tra cứu phổ biến của nhân viên.

#### 3. Timeline triển khai
- **Tuần 1:**
  - Chuẩn hóa bộ dữ liệu văn bản pháp quy và quy chế nội bộ công ty.
  - Xây dựng module Parser bóc tách cấu trúc theo Chương - Điều - Khoản và gán metadata hiệu lực.
  - Triển khai Qdrant Vector Database và BM25 index trên cụm server nội bộ.
- **Tuần 2:**
  - Ghép nối pipeline Hybrid Search và tích hợp mô hình Cross-Encoder Reranker.
  - Xây dựng pipeline đánh giá tự động RAGAS CI/CD và bảng theo dõi chất lượng.
  - Thử nghiệm trên 50 câu hỏi kiểm thử và tinh chỉnh ngưỡng rerank score trước khi đưa lên production.
