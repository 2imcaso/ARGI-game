# Báo cáo kết quả thực nghiệm — Smart Farm AI Robot

Dự án **Smart Farm AI Robot** mô phỏng robot nông nghiệp AGRI-1 khôi phục khu vườn sau bão bằng nhiều nhóm thuật toán trí tuệ nhân tạo. Mỗi khu vực trên bản đồ tương ứng với một nhóm bài toán khác nhau: tìm kiếm không có thông tin, tìm kiếm có heuristic, tìm kiếm cục bộ, tìm kiếm online trong môi trường chưa biết, bài toán thỏa mãn ràng buộc và tìm kiếm đối kháng.

Bản README này được viết lại theo hướng **báo cáo kết quả thực nghiệm**. Phần trọng tâm không chỉ mô tả thuật toán, mà còn tổng hợp thời gian chạy, trạng thái hoàn thành, biểu đồ trực quan theo từng mode và lựa chọn thuật toán tốt nhất đại diện cho mỗi mode.

---

## 1. Mục tiêu báo cáo

Báo cáo nhằm đánh giá kết quả chạy của các thuật toán trong 6 mode của game thông qua ba tiêu chí chính:

| Tiêu chí | Ý nghĩa |
|---|---|
| `algo` | Tên thuật toán được chạy trong từng mode |
| `time` | Thời gian hoàn thành hoặc thời điểm thuật toán bị kẹt, tính bằng giây |
| `success` | Trạng thái chạy: `1` là hoàn thành, `0` là kẹt hoặc thất bại |

Tiêu chí chọn thuật toán tốt nhất trong từng mode là: **ưu tiên thuật toán hoàn thành nhiệm vụ**, sau đó chọn thuật toán có **thời gian thấp nhất**. Riêng Mode 6, nếu hai thuật toán có cùng thời gian, báo cáo ưu tiên thuật toán có ý nghĩa tối ưu hơn về mặt lý thuyết.

---

## 2. Tổng quan các mode và nhóm thuật toán

| Mode | Nhóm thuật toán | Bài toán mô phỏng |
|---|---|---|
| Mode 1 | Uninformed Search | Robot tìm đường khi chưa dùng heuristic |
| Mode 2 | Informed Search | Robot tìm đường bằng heuristic và hàm đánh giá |
| Mode 3 | Local Search | Robot ra quyết định cục bộ, dễ gặp cực trị địa phương |
| Mode 4 | Online Search / Unknown Environment | Robot vừa di chuyển vừa cập nhật môi trường chưa biết |
| Mode 5 | Constraint Satisfaction Problem | Robot gieo cây theo ràng buộc hợp lệ |
| Mode 6 | Adversarial Search | Robot ra quyết định khi có đối thủ và yếu tố ngẫu nhiên |

---

## 3. Dữ liệu thực nghiệm tổng hợp

| Mode | Nhóm | Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| Mode 1 | Uninformed Search | BFS | 34s | Thành công |  |
| Mode 1 | Uninformed Search | DFS | 27s | Thành công | Tốt nhất |
| Mode 1 | Uninformed Search | UCS | 36s | Thành công |  |
| Mode 1 | Uninformed Search | IDS | 30s | Thành công |  |
| Mode 2 | Informed Search | Greedy | 34s | Thành công | Tốt nhất |
| Mode 2 | Informed Search | A* | 37s | Thành công |  |
| Mode 2 | Informed Search | IDA* | 38s | Thành công |  |
| Mode 2 | Informed Search | A*VS | 38s | Thành công |  |
| Mode 3 | Local Search | Hill Climbing | 6s | Kẹt / thất bại |  |
| Mode 3 | Local Search | Local Beam | 29s | Thành công | Tốt nhất |
| Mode 3 | Local Search | SA | 15s | Kẹt / thất bại |  |
| Mode 3 | Local Search | Restart Hill | 30s | Thành công |  |
| Mode 4 | Online Search và môi trường chưa biết | Online A* | 30s | Thành công |  |
| Mode 4 | Online Search và môi trường chưa biết | Online BFS | 39s | Thành công |  |
| Mode 4 | Online Search và môi trường chưa biết | Belief BFS | 37s | Thành công |  |
| Mode 4 | Online Search và môi trường chưa biết | AND-OR | 28s | Thành công | Tốt nhất |
| Mode 5 | Constraint Satisfaction Problem | Backtracking | 65s | Thành công |  |
| Mode 5 | Constraint Satisfaction Problem | Forward Checking | 65s | Thành công |  |
| Mode 5 | Constraint Satisfaction Problem | AC-3 | 60s | Thành công |  |
| Mode 5 | Constraint Satisfaction Problem | Min-conflict | 26s | Thành công | Tốt nhất |
| Mode 6 | Adversarial Search | Minimax | 46s | Thành công |  |
| Mode 6 | Adversarial Search | Alpha-Beta | 46s | Thành công | Tốt nhất |
| Mode 6 | Adversarial Search | Expectimax | 51s | Thành công |  |
| Mode 6 | Adversarial Search | Expectiminimax | 52s | Thành công |  |

---

## 4. Mode 1 — Uninformed Search

Mode 1 gồm các thuật toán tìm kiếm không có thông tin. Tất cả thuật toán đều hoàn thành, cho thấy môi trường của mode này tương đối ổn định. DFS có thời gian thấp nhất nên được chọn làm đại diện tốt nhất của nhóm, dù BFS và IDS thường có tính hệ thống cao hơn trong lý thuyết tìm kiếm.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| BFS | 34s | Thành công |  |
| DFS | 27s | Thành công | Tốt nhất |
| UCS | 36s | Thành công |  |
| IDS | 30s | Thành công |  |

![Biểu đồ Mode 1](assets/charts/mode_1_results.png)

**Thuật toán tốt nhất của Mode 1: `DFS` — 27s.**

---

## 5. Mode 2 — Informed Search

Mode 2 dùng heuristic để định hướng đường đi. Greedy đạt thời gian thấp nhất vì chỉ tập trung vào giá trị heuristic gần nhất, nhưng nhược điểm là có thể không tối ưu toàn cục. A* và A*VS chậm hơn một chút do phải cân bằng giữa chi phí đã đi và ước lượng còn lại.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| Greedy | 34s | Thành công | Tốt nhất |
| A* | 37s | Thành công |  |
| IDA* | 38s | Thành công |  |
| A*VS | 38s | Thành công |  |

![Biểu đồ Mode 2](assets/charts/mode_2_results.png)

**Thuật toán tốt nhất của Mode 2: `Greedy` — 34s.**

---

## 6. Mode 3 — Local Search

Mode 3 thể hiện rõ rủi ro của tìm kiếm cục bộ. Hill Climbing và Simulated Annealing bị kẹt trong lần chạy thực nghiệm, vì vậy không được chọn dù thời gian dừng thấp. Local Beam là thuật toán hoàn thành nhanh nhất trong các thuật toán thành công, còn Random Restart Hill giúp cải thiện khả năng thoát kẹt.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| Hill Climbing | 6s | Kẹt / thất bại |  |
| Local Beam | 29s | Thành công | Tốt nhất |
| SA | 15s | Kẹt / thất bại |  |
| Restart Hill | 30s | Thành công |  |

![Biểu đồ Mode 3](assets/charts/mode_3_results.png)

**Thuật toán tốt nhất của Mode 3: `Local Beam` — 29s.**

---

## 7. Mode 4 — Online Search và môi trường chưa biết

Mode 4 là môi trường chưa biết, robot phải vừa khám phá vừa cập nhật kế hoạch. AND-OR đạt thời gian thấp nhất trong các thuật toán hoàn thành, cho thấy cách xét nhiều kết quả có thể xảy ra phù hợp với môi trường có yếu tố bất định.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| Online A* | 30s | Thành công |  |
| Online BFS | 39s | Thành công |  |
| Belief BFS | 37s | Thành công |  |
| AND-OR | 28s | Thành công | Tốt nhất |

![Biểu đồ Mode 4](assets/charts/mode_4_results.png)

**Thuật toán tốt nhất của Mode 4: `AND-OR` — 28s.**

---

## 8. Mode 5 — Constraint Satisfaction Problem

Mode 5 là bài toán ràng buộc. Min-conflict có thời gian tốt nhất vì chiến lược sửa dần biến đang xung đột phù hợp với bài toán đã có cấu hình gần đúng. Backtracking và Forward Checking hoàn thành nhưng mất nhiều thời gian hơn do phải kiểm tra và quay lui nhiều bước.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| Backtracking | 65s | Thành công |  |
| Forward Checking | 65s | Thành công |  |
| AC-3 | 60s | Thành công |  |
| Min-conflict | 26s | Thành công | Tốt nhất |

![Biểu đồ Mode 5](assets/charts/mode_5_results.png)

**Thuật toán tốt nhất của Mode 5: `Min-conflict` — 26s.**

---

## 9. Mode 6 — Adversarial Search

Mode 6 là bài toán đối kháng. Minimax và Alpha-Beta cùng đạt 46s, nhưng Alpha-Beta được chọn làm đại diện vì vẫn giữ logic quyết định của Minimax trong khi có cơ chế cắt tỉa nhánh không cần xét. Expectimax và Expectiminimax phù hợp hơn khi cần mô phỏng thêm yếu tố xác suất, nhưng thời gian chạy cao hơn.

| Thuật toán | Thời gian | Kết quả | Ghi chú |
| --- | --- | --- | --- |
| Minimax | 46s | Thành công |  |
| Alpha-Beta | 46s | Thành công | Tốt nhất |
| Expectimax | 51s | Thành công |  |
| Expectiminimax | 52s | Thành công |  |

![Biểu đồ Mode 6](assets/charts/mode_6_results.png)

**Thuật toán tốt nhất của Mode 6: `Alpha-Beta` — 46s.**

---

## 10. So sánh thuật toán tốt nhất giữa các mode

Sau khi chọn thuật toán tốt nhất của từng mode, bảng dưới đây dùng để so sánh hiệu quả đại diện giữa các nhóm bài toán.

| Mode | Nhóm | Thuật toán tốt nhất | Thời gian | Kết quả |
| --- | --- | --- | --- | --- |
| Mode 1 | Uninformed Search | DFS | 27s | Thành công |
| Mode 2 | Informed Search | Greedy | 34s | Thành công |
| Mode 3 | Local Search | Local Beam | 29s | Thành công |
| Mode 4 | Online Search và môi trường chưa biết | AND-OR | 28s | Thành công |
| Mode 5 | Constraint Satisfaction Problem | Min-conflict | 26s | Thành công |
| Mode 6 | Adversarial Search | Alpha-Beta | 46s | Thành công |

![So sánh thuật toán tốt nhất từng mode](assets/charts/best_by_mode_comparison.png)

Nhìn chung, **Min-conflict ở Mode 5** là thuật toán tốt nhất toàn bộ thực nghiệm theo tiêu chí thời gian hoàn thành, với kết quả 26s. Tuy nhiên, kết quả này không có nghĩa Min-conflict mạnh hơn mọi thuật toán khác trong mọi bài toán, mà chỉ cho thấy nó phù hợp nhất với bài toán ràng buộc trong môi trường demo đã xây dựng.

---

## 11. Nhận xét tổng hợp

Từ kết quả thực nghiệm có thể rút ra một số nhận xét chính:

- Ở nhóm tìm kiếm cơ bản, DFS nhanh nhất trong lần chạy này, nhưng BFS và IDS vẫn có giá trị minh họa tốt vì cách duyệt trạng thái rõ ràng hơn.
- Ở nhóm tìm kiếm có heuristic, Greedy nhanh nhất nhưng A* có cơ sở tốt hơn khi cần cân bằng giữa đường đã đi và chi phí ước lượng.
- Ở nhóm Local Search, kết quả cho thấy các thuật toán cục bộ dễ bị kẹt nếu hàm đánh giá hoặc điểm khởi đầu không tốt.
- Ở môi trường chưa biết, AND-OR cho kết quả tốt nhất vì phù hợp với bài toán có nhiều khả năng xảy ra trong tương lai.
- Ở bài toán CSP, Min-conflict nổi bật do xử lý trực tiếp các xung đột thay vì thử toàn bộ tổ hợp từ đầu.
- Ở bài toán đối kháng, Alpha-Beta là lựa chọn đại diện hợp lý vì đạt cùng thời gian với Minimax nhưng có cơ chế tối ưu bằng cắt tỉa.

---

## 12. Kết luận

Dự án đã triển khai thành công 6 nhóm thuật toán AI trong cùng một môi trường game Smart Farm. Thay vì chỉ trình bày lý thuyết, hệ thống cho phép quan sát trực tiếp cách robot chọn mục tiêu, tìm đường, xử lý ràng buộc, cập nhật môi trường và ra quyết định khi có đối thủ.

Kết quả thực nghiệm cho thấy không có thuật toán nào tốt nhất cho mọi tình huống. Mỗi nhóm thuật toán phù hợp với một dạng bài toán riêng: tìm kiếm đường đi, tối ưu heuristic, tìm kiếm cục bộ, môi trường chưa biết, bài toán ràng buộc hoặc đối kháng. Do đó, giá trị chính của dự án là minh họa được sự khác biệt trong cách ra quyết định của từng nhóm thuật toán và tạo cơ sở để so sánh trực quan thông qua thời gian chạy, trạng thái hoàn thành và biểu đồ kết quả.

---

## 13. Hướng phát triển

Trong tương lai, dự án có thể được mở rộng theo các hướng sau:

- Bổ sung thêm chỉ số đánh giá như số node đã mở rộng, độ dài đường đi và số lần quay lui.
- Thêm chức năng xuất thống kê tự động ra CSV sau mỗi lần chạy.
- Cho phép người dùng tự tạo bản đồ và thay đổi độ khó của từng mode.
- Mở rộng Mode 6 với nhiều đối thủ, nhiều sự kiện xác suất và độ sâu tìm kiếm lớn hơn.
- Tích hợp dashboard trực quan trong game để so sánh thuật toán ngay sau khi chạy.
