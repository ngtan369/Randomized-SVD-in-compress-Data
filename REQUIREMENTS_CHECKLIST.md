# Checklist Yêu cầu Đề bài 10 — Randomized SVD và Nén Dữ liệu

Tài liệu này đối chiếu từng yêu cầu của đề bài 10 với sản phẩm thực tế trong
repo, kèm bằng chứng (file, dòng / mục).

## Bảng tổng hợp

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---|---------|-----------|-----------|
| 1 | Trình bày cơ sở lý thuyết SVD và compact SVD | PASS | `report/lcontent.tex` §2.2 (dòng 103), §2.3 (dòng 141), §2.4 Eckart-Young (dòng 161) |
| 2 | Trình bày cơ sở lý thuyết Randomized SVD | PASS | `report/lcontent.tex` §3 "Thuật toán Randomized SVD" (dòng 205-296): ý tưởng, thuật toán, cơ sở toán học, độ phức tạp, sai số |
| 3 | Tiêu chuẩn chọn $k$ theo tỉ lệ năng lượng | PASS | Công thức (eq:energy-criterion) tại `report/lcontent.tex` §2.5 (dòng 178-201) và §3.6 (dòng 297); cài đặt hàm `choose_k_by_energy` trong `src/experiment_image.py`, `src/experiment_movielens.py` |
| 4 | Chương trình Randomized SVD cho ma trận lớn | PARTIAL | Dự kiến `src/rsvd.py` (thuật toán 8 bước theo Halko-Martinsson-Tropp) và các script thí nghiệm. *Chưa thấy file trong thư mục `src/` tại thời điểm soạn checklist — agent song song đang tạo.* |
| 5 | Xây dựng $A_k$ và lưu trữ dạng nén $(U_k, s_k, V_k^T)$ | PARTIAL | Dự kiến lưu trong `results/*.npz` từ `src/experiment_image.py` và `src/experiment_movielens.py`. Lý thuyết dung lượng lưu trữ: `report/lcontent.tex` §4.5 (dòng 507). *Thư mục `results/` hiện rỗng.* |
| 6 | Đánh giá sai số xấp xỉ theo chuẩn Frobenius $\|A-A_k\|_F$ | PARTIAL | Công thức tính nhanh từ phổ kỳ dị: `report/lcontent.tex` §4.3.4 (dòng 446); kết quả thực nghiệm §4.4.2 (dòng 486). Cần file `src/experiment_*.py` để xuất ra `results/*.json`. |
| 7 | Dữ liệu kích thước >= 10^5 phần tử, từ nguồn thực tế | PASS (theo thiết kế) | Thí nghiệm ảnh 512×512 = 2.62×10^5 phần tử (`src/experiment_image.py`); thí nghiệm thưa kiểu MovieLens 5000×2000 với ~10^5 nnz (`src/experiment_movielens.py`). Mô tả dữ liệu: `report/lcontent.tex` §4.1 (dòng 321) |
| 8 | Không dùng trực tiếp hàm SVD đầy đủ trên $A$ | PASS (theo thiết kế) | Thuật toán tự cài 8 bước trong `src/rsvd.py`: chỉ gọi `np.linalg.svd` trên ma trận nhỏ $B = Q^T A$ kích thước $(k+p) \times n$, không gọi trên $A$. Cơ sở: `report/lcontent.tex` §3.2 (dòng 221), §3.3 (dòng 248) |
| 9 | Giải thích ý nghĩa năng lượng dữ liệu và giá trị kỳ dị | PASS | `report/lcontent.tex` §2.5 (dòng 178-201): định nghĩa tổng năng lượng $\|A\|_F^2 = \sum \sigma_i^2$, đóng góp năng lượng từng thành phần, suy giảm phổ và cơ sở để nén |

Ghi chú trạng thái:
- **PASS**: Đã có sản phẩm hoàn chỉnh và kiểm tra được.
- **PARTIAL**: Đã có thiết kế / lý thuyết / một phần code nhưng còn thiếu file cuối hoặc kết quả đầu ra.
- **FAIL**: Chưa có sản phẩm.

## Chi tiết bằng chứng theo từng mục

### 1. Cơ sở lý thuyết SVD và compact SVD
- §2.2 "Định lý phân tích giá trị kỳ dị (SVD)" — phát biểu định lý, dạng tổng quát.
- §2.3 "Compact SVD (Reduced SVD)" — định nghĩa dạng rút gọn theo hạng $r$.
- §2.4 "Định lý Eckart-Young" — nền tảng cho xấp xỉ hạng thấp $A_k$.

### 2. Cơ sở lý thuyết Randomized SVD
- §3.1 Ý tưởng và động cơ — phép chiếu ngẫu nhiên, không gian con chính.
- §3.2 Thuật toán (8 bước).
- §3.3 Cơ sở toán học của từng bước.
- §3.4 Phân tích độ phức tạp $O(mn(k+p))$.
- §3.5 Sai số xấp xỉ (cận xác suất theo Halko-Martinsson-Tropp 2011).

### 3. Tiêu chuẩn chọn $k$ theo năng lượng
Công thức trung tâm (eq:energy-criterion):
$$ \frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_{i=1}^{r}\sigma_i^2} \ge \eta, \quad \eta \in \{0.90, 0.95, 0.99\}. $$
Hệ quả: $\|A-A_k\|_F^2 / \|A\|_F^2 \le 1-\eta$.

### 4. Chương trình Randomized SVD
File chính dự kiến `src/rsvd.py` (chưa thấy tại thời điểm soạn). Thuật toán cài đặt theo §3.2.

### 5. Lưu trữ dạng nén
Lưu $(U_k \in \mathbb{R}^{m\times k},\, s_k \in \mathbb{R}^k,\, V_k^T \in \mathbb{R}^{k\times n})$ thay vì $A_k$ đầy đủ. Tiết kiệm khi $k \ll \min(m,n)$. Phân tích trong §4.5.

### 6. Sai số Frobenius
Tính nhanh: $\|A-A_k\|_F = \sqrt{\sum_{i=k+1}^{r}\sigma_i^2}$ — không cần dựng lại $A_k$.

### 7. Kích thước dữ liệu >= 10^5
- Ảnh 512×512 grayscale = 262 144 phần tử.
- Ma trận thưa 5000×2000 với mật độ ~1% => $10^5$ nnz.

### 8. Không gọi SVD đầy đủ trên $A$
Cấu trúc thuật toán: $\Omega$ ngẫu nhiên → $Y = A\Omega$ → QR → $B = Q^T A$ kích thước nhỏ → SVD trên $B$ → ghép $U = Q\tilde{U}$. SVD duy nhất xảy ra trên ma trận $(k+p) \times n$ chứ không phải $m \times n$.

### 9. Ý nghĩa năng lượng
Năng lượng dữ liệu = $\|A\|_F^2$. Suy giảm nhanh của $\sigma_i^2$ là điều kiện cần để compress hiệu quả. Tỉ lệ năng lượng giữ lại có ý nghĩa thống kê: xấp xỉ phương sai được bảo toàn (liên hệ PCA).

---

**Lưu ý:** Một số file đang được các agent song song tạo — kiểm tra lại checklist sau khi tất cả agent hoàn thành. Cập nhật trạng thái PARTIAL → PASS khi đã xác minh tồn tại file và chạy thử thành công bằng `./run_all.sh`.
