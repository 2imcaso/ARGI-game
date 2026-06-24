# Smart Farm AI Robot - 6 ngay AI

Game dung cung mot nong trai Pydew, nhung 6 man cu da duoc thay bang 6 ngay sau bao cua AGRI-1. Moi ngay la mot khu dat rieng trong nong trai, gan voi mot nhom thuat toan AI khac nhau.

## Tong quan 6 khu

| Ngay | Khu | Tinh huong | Nhom bai toan | Thuat toan |
| --- | --- | --- | --- | --- |
| 1 | Vuon truoc trai | Bat dau lai tu dau, robot chua co ban do va tim khu dat gan nhat | Uninformed Search | BFS |
| 2 | Loi vao nha kho | Manh vo nha kho/da chan duong, robot da biet dich va vat can | Informed Search | A* |
| 3 | Vuon cay phia dong | Qua nhieu cay can cuu, phai chon cay uu tien trong thoi gian ngan | Local Search | Local Beam / Hill Climbing / Annealing |
| 4 | Rescue the Foggy Garden | Xu ly toan bo khu vuon bang mot agent doc lap | Rescue Agents | Belief State / Goal-Based / AND-OR |
| 5 | O quy hoach trung tam | Quy hoach lai vuon, ngo va ca chua khong duoc ke nhau | Constraint Satisfaction | CSP Backtracking |
| 6 | Hang cay can bao ve | Ke pha hoai tranh den cac o cay de pha | Adversarial Search | Minimax / Alpha-Beta / Expectimax / Expectiminimax |

## Dieu khien

- `1`: Ngay 1 - BFS
- `2`: Ngay 2 - A*
- `3`: Ngay 3 - Local Search
- `4`: Man 4 - Rescue Agents
- `5`: Ngay 5 - CSP Backtracking
- `6`: Ngay 6 - Minimax / Alpha-Beta / Expectimax / Expectiminimax

## Chi tiet tung ngay

### Ngay 1 - Khu vuon truoc trai - BFS

AGRI-1 vua khoi dong sau bao, chua co ban do va khong biet nen uu tien huong nao. Robot duyet deu theo tung lop de tim den khu dat gan nhat va bat dau cuoc.

### Ngay 2 - Khu loi vao nha kho - A*

Robot da biet bo cuc nong trai, nhung manh vo nha kho tao vat can tren duong. A* dung `f(n)=g(n)+h(n)` va heuristic Manhattan de di vong ngan hon thay vi duyet mu.

### Ngay 3 - Khu vuon cay phia dong - Local Search

Nhieu cay co muc do kho/heo khac nhau. Robot dung cac bien the local search nhu local beam, hill climbing va simulated annealing de toi uu thu tu xu ly toan bo cay con lai, roi thuc hien muc tieu dau tien trong ke hoach tot nhat.

### Ngay 4 - Khu bai dat suong mu - Online Search

Robot chi thay cac o sat ben canh va phai den cay cuu ho o goc xa. Hai buc tuong da an tao ngo cut; khi phat hien block moi, robot cap nhat belief/map va re-plan thay vi dung mot ke hoach tinh tu dau.

### Ngay 5 - Khu quy hoach trung tam - CSP Backtracking

Ong Minh muon trong lai theo rang buoc: corn va tomato khong duoc nam ke nhau. Robot dung backtracking de gan crop cho tung o truoc khi gieo.

### Ngay 6 - Khu hang cay can bao ve - Adversarial Search

Mot ke pha hoai tranh den cac o cay de lam hong cay vua trong. AGRI-1 co the dung Minimax, Alpha-Beta, Expectimax hoac Expectiminimax de chon o can bao ve theo mo hinh doi thu/xac suat khac nhau.

Mode 6 chi co mot hanh dong la di sua cay, nhung 25 cay co tinh trang, gia tri va rui ro khac nhau. Khong co quy trinh cuoc/gieo/tuoi.

| Tinh trang | Ky hieu | Diem | Rui ro |
| --- | --- | ---: | ---: |
| Cay khoe | K | 5 | 10% |
| Cay kho | H | 10 | 20% |
| Cay sau benh | S | 20 | 40% |
| Cay nguy kich | N | 30 | 70% |
| Cay quy | Q | 50 | 50% |

Layout 5x5:

```text
K K H K K
K H S H K
H S Q S H
K H S H K
K K H K K
```

Ca bon thuat toan dung chung ham danh gia:

`score = tong_gia_tri_cay_song - 1.5 * gia_tri_bi_pha + 0.5 * gia_tri_da_sua`

- Minimax: cay `MAX -> MIN`, ke pha hoai choi toi uu.
- Alpha-Beta: cung cay va cung ket qua Minimax, chi cat bot nhanh khi `beta <= alpha`.
- Expectimax: cay `MAX -> CHANCE`, khong co doi thu thong minh; sau benh, mua da va han han xay ra theo xac suat.
- Expectiminimax: cay `MAX -> MIN -> CHANCE`, ket hop ke pha hoai va thoi tiet ngau nhien.

Can bang gameplay Mode 6:

- AGRI-1 va qua di cung toc do, chi theo 4 huong tren luoi.
- Hai ben dung A* tren cung ban do vat can.
- Bao ve va pha cay deu mat 1.5 giay.
- Vi tri xuat phat doi xung qua hai mep vuon.
- Moi vong chi ket thuc khi ca AGRI-1 va doi thu/su kien da hoan tat.
- HUD cap nhat theo tung decision, tung buoc di, luc sua va luc qua pha; moi thuat toan hien dung Depth/Utility, Alpha-Beta-Pruned hoac Expected Utility/Probability cua no.

## File chinh

- `code/algorithms.py`: cac thuat toan rieng: BFS, A*, Local Search, Online vision update, CSP Backtracking, Minimax.
- `code/ai_controller.py`: dieu phoi robot, animation, panel thong tin va visualization.
- `code/level.py`: cau hinh 6 khu moi, spawn, farm tiles, obstacles, hidden blocks va enemy spawn.
- `code/main.py`: vong lap Pygame va phim 1-6 de doi khu.
