# Đề tài 10 — Randomized SVD và Ứng dụng trong Nén Dữ liệu

## Thông tin nhóm
Nhóm L05 — GV. Nguyễn Hữu Hiệp

Thành viên (9):
- 2213539 — Phạm Quốc Toàn
- 2213496 — Nguyễn Quốc Tín
- 2213299 — Nguyễn Trường Thịnh
- 2213289 — Nguyễn Hữu Thịnh
- 2213228 — Hồ Viết Thiên
- 2213120 — Trương Hữu Thái
- 2213080 — Trương Hồng Tấn
- 2213063 — Nguyễn Trung Tân
- 2212922 — Nguyễn Quang Sáng

## Cấu trúc dự án
- `src/rsvd.py` — Cài đặt Randomized SVD từ đầu (thuật toán 8 bước, không gọi `np.linalg.svd` trên ma trận đầy đủ).
- `src/test_rsvd.py` — Unit tests cho các thành phần của Randomized SVD.
- `src/experiment_image.py` — Thí nghiệm nén ảnh (dữ liệu dày, kích thước >= 10^5 phần tử).
- `src/experiment_movielens.py` — Thí nghiệm trên ma trận thưa kiểu MovieLens.
- `src/benchmark.py` — So sánh hiệu năng Randomized SVD với SVD truyền thống / truncated SVD.
- `svd_compressData.ipynb` — Notebook gốc (đã đánh dấu **deprecated**, dùng để tham khảo).
- `data/` — Dữ liệu nguồn cho các thí nghiệm.
- `results/` — Kết quả JSON, ma trận xấp xỉ hạng thấp được lưu nén `(U_k, s_k, V_k^T)`.
- `report/` — Mã nguồn LaTeX của báo cáo.
- `report/main.pdf` — Báo cáo PDF cuối cùng (sản phẩm chính).

## Cài đặt môi trường
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy thí nghiệm
Chạy tất cả thí nghiệm theo thứ tự và build báo cáo:
```bash
./run_all.sh
```

Hoặc chạy lần lượt từng script:
```bash
python src/test_rsvd.py
python src/experiment_image.py
python src/experiment_movielens.py
python src/benchmark.py
```

## Build report
```bash
cd report
pdflatex main.tex
pdflatex main.tex   # chạy lần 2 để mục lục / tham chiếu ổn định
```
Tệp xuất ra: `report/main.pdf`.

## Đề bài 10 — yêu cầu thực hiện
Đề bài 10 yêu cầu 6 nội dung chính và 3 ràng buộc phụ. Trạng thái chi tiết và bằng chứng (file, dòng) xem trong:

[REQUIREMENTS_CHECKLIST.md](REQUIREMENTS_CHECKLIST.md)

Tóm tắt:
1. Trình bày cơ sở lý thuyết SVD và compact SVD.
2. Trình bày cơ sở lý thuyết Randomized SVD.
3. Xây dựng tiêu chuẩn chọn $k$ theo tỉ lệ năng lượng.
4. Viết chương trình Randomized SVD cho ma trận dữ liệu kích thước lớn.
5. Xây dựng ma trận xấp xỉ hạng thấp $A_k$ và lưu trữ dạng nén.
6. Đánh giá sai số xấp xỉ theo chuẩn Frobenius $\|A - A_k\|_F$.
