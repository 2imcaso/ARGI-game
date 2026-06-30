# Smart Farm AI Robot

Đồ án mô phỏng robot nông nghiệp **AGRI-1** khôi phục khu vườn sau bão bằng các thuật toán Trí tuệ nhân tạo. Game được xây dựng bằng Python và Pygame, trong đó mỗi khu vực trên bản đồ đại diện cho một nhóm thuật toán khác nhau.

Robot không chỉ đi theo đường có sẵn mà phải tự chọn mục tiêu, tìm đường, cập nhật trạng thái môi trường và xử lý nhiệm vụ. Nhờ cách mô phỏng bằng bản đồ lưới, quá trình hoạt động của từng thuật toán có thể được quan sát trực tiếp qua đường đi, node đã duyệt, trạng thái cây và các hiệu ứng trong game.

---

## 1. Mục tiêu dự án

Dự án được thực hiện nhằm:

- Trực quan hóa các thuật toán AI trong một môi trường game dễ quan sát.
- So sánh cách robot ra quyết định giữa các nhóm thuật toán khác nhau.
- Mô phỏng đầy đủ các dạng bài toán: tìm kiếm đường đi, tìm kiếm cục bộ, môi trường chưa biết, bài toán ràng buộc và tìm kiếm đối kháng.
- Xây dựng một sản phẩm hoàn chỉnh có giao diện, bản đồ, nhân vật, hiệu ứng và ảnh GIF minh họa kết quả chạy.

---

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ lập trình | Python |
| Thư viện game | Pygame |
| Bản đồ | Tiled Map Editor, TMX, pytmx |
| Kiểu bản đồ | Grid map |
| Heuristic chính | Manhattan Distance |
| Trực quan hóa | HUD, đường đi, node đã duyệt, fog-of-war, hiệu ứng đối thủ, GIF |

---

## 3. Cấu trúc thư mục

```text
Smart-Farm-AI-Robot/
│
├── main.py                  # File chính khởi chạy game
├── level.py                 # Quản lý bản đồ, mode và khu vực demo
├── ai_controller.py         # Điều khiển robot AI và state machine
├── algorithms.py            # Cài đặt thuật toán tìm kiếm, CSP và đối kháng
├── design_tokens.py         # Màu sắc và token hiển thị giao diện
│
├── data/                    # File bản đồ TMX
├── graphics/                # Tài nguyên hình ảnh trong game
├── audio/                   # Âm thanh nền và hiệu ứng
├── gif/                     # GIF minh họa thuật toán
│   ├── A-ezgif.com-video-to-gif-converter.gif
│   ├── A_v2-ezgif.com-video-to-gif-converter.gif
│   ├── BFS-ezgif.com-video-to-gif-converter.gif
│   ├── DFS-ezgif.com-video-to-gif-converter.gif
│   ├── Greedy-ezgif.com-video-to-gif-converter.gif
│   ├── IDS-ezgif.com-video-to-gif-converter.gif
│   ├── IDSA-ezgif.com-video-to-gif-converter.gif
│   ├── UCS-ezgif.com-video-to-gif-converter.gif
│   ├── ac3.gif
│   ├── alphabeta.gif
│   ├── and-or.gif
│   ├── backtrack.gif
│   ├── belief bfs.gif
│   ├── exceptimax.gif
│   ├── exceptiminmax.gif
│   ├── forward.gif
│   ├── hillclimping-ezgif.com-video-to-gif-converter.gif
│   ├── localbeam1-ezgif.com-video-to-gif-converter.gif
│   ├── minconflit.gif
│   ├── minimax.gif
│   ├── online astar.gif
│   ├── online bfs.gif
│   ├── randomrestart-ezgif.com-video-to-gif-converter.gif
│   └── simulatedannealing-ezgif.com-video-to-gif-converter.gif
│
└── README.md
```

---

## 4. Cách chạy chương trình

Cài đặt thư viện cần thiết:

```bash
pip install pygame pytmx
```

Chạy game:

```bash
python main.py
```

Sau khi chạy, cửa sổ **Smart Farm AI Robot** sẽ hiển thị. Người dùng có thể chọn từng khu vực, đổi thuật toán và quan sát robot tự xử lý nhiệm vụ.

---

## 5. Hướng dẫn điều khiển

| Phím / thao tác | Chức năng |
|---|---|
| `1` đến `6` | Chuyển khu vực/mode |
| `Q` | Chọn thuật toán trước trong cùng nhóm |
| `E` | Chọn thuật toán tiếp theo trong cùng nhóm |
| Chuột trái | Chọn thuật toán, Start, Pause hoặc Reset trên panel |
| `Enter` ở Mode 6 | Chuyển sang màn hình kết thúc |

---

## 6. Mô hình bài toán

Trong game, bản đồ được biểu diễn như một không gian trạng thái dạng lưới. Mỗi trạng thái có thể gồm vị trí robot, danh sách ô đã xử lý, danh sách ô còn lại, vật cản, đường đi hiện tại và các trạng thái riêng của từng mode.

Luồng xử lý chung của robot:

```text
Chọn mục tiêu → Tìm đường → Di chuyển → Xử lý cây/ô đất → Cập nhật trạng thái
```

Tùy thuật toán đang được chọn, robot sẽ có cách duyệt node, chọn mục tiêu và lập kế hoạch khác nhau.

---

## 7. Các nhóm thuật toán đã triển khai

### 7.1. Mode 1 — Uninformed Search

Mode 1 mô phỏng nhóm thuật toán tìm kiếm không có thông tin. Robot chưa dùng heuristic mà chỉ dựa vào trạng thái ban đầu, trạng thái đích và các hành động hợp lệ.

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| BFS | Duyệt theo từng tầng, tìm đường ngắn nhất nếu chi phí mỗi bước bằng nhau | ![BFS](gif/BFS-ezgif.com-video-to-gif-converter.gif) |
| DFS | Duyệt sâu theo một nhánh trước khi quay lại | ![DFS](gif/DFS-ezgif.com-video-to-gif-converter.gif) |
| UCS | Ưu tiên đường đi có tổng chi phí thấp nhất | ![UCS](gif/UCS-ezgif.com-video-to-gif-converter.gif) |
| IDS | Lặp DFS với giới hạn độ sâu tăng dần | ![IDS](gif/IDS-ezgif.com-video-to-gif-converter.gif) |

Nhóm thuật toán này phù hợp để minh họa nền tảng của tìm kiếm trạng thái. BFS và IDS có tính ổn định cao hơn, DFS tiết kiệm bộ nhớ hơn, còn UCS có ý nghĩa rõ khi bản đồ có chi phí di chuyển khác nhau.

---

### 7.2. Mode 2 — Informed Search

Mode 2 sử dụng heuristic để robot chọn đường đi tốt hơn. Heuristic chính là khoảng cách Manhattan kết hợp với độ ưu tiên của cây cần xử lý.

Công thức sử dụng:

```text
h(n) = Manhattan(n, target) + 2 * condition(target)
f(n) = g(n) + h(n)
```

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| Greedy | Chọn node có `h(n)` nhỏ nhất | ![Greedy](gif/Greedy-ezgif.com-video-to-gif-converter.gif) |
| A* | Kết hợp chi phí đã đi `g(n)` và heuristic `h(n)` | ![A*](gif/A-ezgif.com-video-to-gif-converter.gif) |
| A*_v2 | A* có thêm chi phí phạt khi rẽ hướng | ![A*_v2](gif/A_v2-ezgif.com-video-to-gif-converter.gif) |
| IDSA | Tìm kiếm theo ngưỡng `f(n)` tăng dần | ![IDSA](gif/IDSA-ezgif.com-video-to-gif-converter.gif) |

So với Mode 1, nhóm này giúp robot di chuyển có định hướng hơn. Greedy phản ứng nhanh nhưng có thể chọn đường chưa tối ưu. A* cân bằng tốt hơn vì xét cả chi phí đã đi và ước lượng còn lại.

---

### 7.3. Mode 3 — Local Search

Mode 3 mô phỏng nhóm thuật toán tìm kiếm cục bộ. Robot không duyệt toàn bộ không gian trạng thái mà tập trung cải thiện lựa chọn hiện tại hoặc một nhóm ứng viên gần nhất.

Điểm đánh giá được xây dựng dựa trên loại cây và khoảng cách Manhattan:

```text
score = điểm loại cây + Manhattan(current, tile) - turn_penalty
```

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| Hill Climbing | Chọn trạng thái lân cận tốt hơn hiện tại | ![Hill Climbing](gif/hillclimping-ezgif.com-video-to-gif-converter.gif) |
| Local Beam | Giữ nhiều ứng viên tốt nhất cùng lúc | ![Local Beam](gif/localbeam1-ezgif.com-video-to-gif-converter.gif) |
| Simulated Annealing | Đôi khi chấp nhận bước xấu để thoát kẹt cục bộ | ![Simulated Annealing](gif/simulatedannealing-ezgif.com-video-to-gif-converter.gif) |
| Random Restart Hill | Khởi động lại khi bị kẹt ở cực trị cục bộ | ![Random Restart](gif/randomrestart-ezgif.com-video-to-gif-converter.gif) |

Nhóm Local Search cho thấy robot có thể ra quyết định nhanh trong khu vực nhỏ, nhưng cũng dễ bị ảnh hưởng bởi cực trị cục bộ nếu chỉ nhìn vào lựa chọn gần nhất.

---

### 7.4. Mode 4 — Online Search và môi trường chưa biết

Mode 4 đặt robot vào khu vực có sương mù và vật cản ẩn. Robot không biết toàn bộ bản đồ ngay từ đầu mà phải vừa di chuyển, vừa quan sát, vừa cập nhật lại kế hoạch.

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| Online A* | Lập kế hoạch lại khi có thông tin mới | ![Online A*](gif/online%20astar.gif) |
| Online BFS | Duyệt theo lớp trong môi trường được khám phá dần | ![Online BFS](gif/online%20bfs.gif) |
| Belief-State BFS | Tìm kiếm trên tập trạng thái có thể xảy ra | ![Belief BFS](gif/belief%20bfs.gif) |
| AND-OR Search | Lập kế hoạch cho nhiều kết quả có thể xảy ra | ![AND-OR](gif/and-or.gif) |

Mode này thể hiện rõ sự khác biệt giữa môi trường biết trước và môi trường chưa biết. Robot phải liên tục cập nhật thông tin thay vì chỉ chạy một kế hoạch cố định từ đầu đến cuối.

---

### 7.5. Mode 5 — Constraint Satisfaction Problem

Mode 5 mô phỏng bài toán thỏa mãn ràng buộc. Robot cần gieo cây trên lưới sao cho các ô liền kề không vi phạm điều kiện đã đặt ra.

Các loại cây được sử dụng:

```text
corn, tomato, wheat, carrot
```

Một số ràng buộc chính:

- Hai ô kề nhau không được trồng cùng loại cây.
- Cặp `corn - tomato` không được đứng cạnh nhau.
- Cặp `wheat - carrot` không được đứng cạnh nhau.

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| Backtracking | Thử gán giá trị và quay lui khi vi phạm | ![Backtracking](gif/backtrack.gif) |
| Forward Checking | Loại trước giá trị không còn hợp lệ ở biến lân cận | ![Forward Checking](gif/forward.gif) |
| AC-3 | Duy trì tính nhất quán cung giữa các biến | ![AC-3](gif/ac3.gif) |
| Min Conflict | Sửa dần các biến đang gây xung đột | ![Min Conflict](gif/minconflit.gif) |

Nhóm CSP giúp minh họa bài toán không chỉ cần tìm đường đi mà còn cần tìm một cách sắp xếp thỏa tất cả điều kiện.

---

### 7.6. Mode 6 — Adversarial Search

Mode 6 là khu vực đối kháng. Robot AGRI-1 đóng vai trò **MAX**, có nhiệm vụ bảo vệ và sửa cây. Đối thủ đóng vai trò **MIN**, cố gắng phá cây. Một số thuật toán còn có thêm yếu tố ngẫu nhiên như sét đánh hoặc cột thu lôi.

| Thuật toán | Ý tưởng chính | GIF minh họa |
|---|---|---|
| Minimax | MAX chọn nước đi tốt nhất, giả định MIN cũng tối ưu | ![Minimax](gif/minimax.gif) |
| Alpha-Beta | Tối ưu Minimax bằng cách cắt tỉa nhánh không cần xét | ![Alpha-Beta](gif/alphabeta.gif) |
| Expectimax | Thêm node xác suất để mô phỏng sự kiện ngẫu nhiên | ![Expectimax](gif/exceptimax.gif) |
| Expectiminimax | Kết hợp MAX, MIN và CHANCE trong cùng cây quyết định | ![Expectiminimax](gif/exceptiminmax.gif) |

Mode này thể hiện bài toán ra quyết định khi có đối thủ. Robot không chỉ cần chọn cây gần nhất, mà phải cân nhắc nước đi của đối phương và rủi ro từ các sự kiện ngẫu nhiên.

---

## 8. Kết quả đạt được

Dự án đã hoàn thành một game demo có thể chạy trực tiếp, gồm 6 khu vực tương ứng với 6 nhóm thuật toán AI. Mỗi nhóm thuật toán được gắn với một tình huống cụ thể trong game để người xem dễ hiểu vai trò của thuật toán.

Các kết quả chính:

- Xây dựng được bản đồ nông trại gồm nhiều khu vực demo riêng biệt.
- Tích hợp robot tự động di chuyển, tìm mục tiêu và xử lý nhiệm vụ.
- Cài đặt các nhóm thuật toán: Uninformed Search, Informed Search, Local Search, Online Search, CSP và Adversarial Search.
- Hiển thị được đường đi, node đã duyệt, thông số `f(n)`, `g(n)`, `h(n)`, trạng thái belief, bước backtracking và quyết định đối kháng.
- Tạo GIF minh họa cho từng thuật toán để trình bày trong README và báo cáo.

---

## 9. Hướng phát triển

Trong tương lai, dự án có thể mở rộng theo các hướng sau:

- Bổ sung thêm thuật toán mới như D*, LRTA*, Genetic Algorithm hoặc Monte Carlo Tree Search.
- Cho phép người dùng tự tạo bản đồ và tự đặt vật cản.
- Thêm chế độ so sánh thời gian chạy, số node mở rộng và độ dài đường đi giữa các thuật toán.
- Cải thiện giao diện chọn mode, bảng thống kê và phần giải thích thuật toán trong game.
- Mở rộng Mode 6 với nhiều loại đối thủ và nhiều dạng sự kiện ngẫu nhiên hơn.

---

## 10. Nguồn tham khảo và credit

Dự án có sử dụng và tùy biến một số tài nguyên đồ họa pixel-art từ **Sprout Lands Asset Pack** của **Cup Nooble**. Các tài nguyên này được dùng cho mục đích học tập, minh họa giao diện và xây dựng môi trường game.

Nguồn của tác giả:

- Cup Nooble: https://cupnooble.carrd.co/
- Cup Nooble YouTube: https://www.youtube.com/@Cup_Nooble
- Sprout Lands Asset Pack: https://cupnooble.itch.io/sprout-lands-asset-pack

Trích nguồn ngắn gọn:

```text
Cup Nooble. (n.d.). Sprout Lands Asset Pack. itch.io.
https://cupnooble.itch.io/sprout-lands-asset-pack

Cup Nooble. (n.d.). Cup Nooble official links.
https://cupnooble.carrd.co/
```

Bản quyền gốc của các asset thuộc về tác giả Cup Nooble. Dự án này chỉ sử dụng tài nguyên trong phạm vi học tập và báo cáo môn học. Nếu phát triển hoặc phát hành ở phạm vi thương mại, cần kiểm tra lại điều khoản sử dụng chính thức của asset pack.

---

