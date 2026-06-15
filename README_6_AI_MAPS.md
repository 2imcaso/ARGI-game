# Smart Farm AI Robot - 6 ngay AI

Game dung cung mot nong trai Pydew, nhung 6 man cu da duoc thay bang 6 ngay sau bao cua AGRI-1. Moi ngay la mot khu dat rieng trong nong trai, gan voi mot nhom thuat toan AI khac nhau.

## Tong quan 6 khu

| Ngay | Khu | Tinh huong | Nhom bai toan | Thuat toan |
| --- | --- | --- | --- | --- |
| 1 | Vuon truoc trai | Bat dau lai tu dau, robot chua co ban do va tim khu dat gan nhat | Uninformed Search | BFS |
| 2 | Loi vao nha kho | Manh vo nha kho/da chan duong, robot da biet dich va vat can | Informed Search | A* |
| 3 | Vuon cay phia dong | Qua nhieu cay can cuu, phai chon cay uu tien trong thoi gian ngan | Local Search | Local Beam / Hill Climbing / Annealing |
| 4 | Bai dat suong mu | Suong mu sau bao, chi thay 3 o xung quanh va vat can an | Online/Partial Observability | Online Search |
| 5 | O quy hoach trung tam | Quy hoach lai vuon, ngo va ca chua khong duoc ke nhau | Constraint Satisfaction | CSP Backtracking |
| 6 | Hang cay can bao ve | Ke pha hoai tranh den cac o cay de pha | Adversarial Search | Minimax / Alpha-Beta |

## Dieu khien

- `1`: Ngay 1 - BFS
- `2`: Ngay 2 - A*
- `3`: Ngay 3 - Local Search
- `4`: Ngay 4 - Online Search
- `5`: Ngay 5 - CSP Backtracking
- `6`: Ngay 6 - Minimax / Alpha-Beta

## Chi tiet tung ngay

### Ngay 1 - Khu vuon truoc trai - BFS

AGRI-1 vua khoi dong sau bao, chua co ban do va khong biet nen uu tien huong nao. Robot duyet deu theo tung lop de tim den khu dat gan nhat va bat dau cuoc.

### Ngay 2 - Khu loi vao nha kho - A*

Robot da biet bo cuc nong trai, nhung manh vo nha kho tao vat can tren duong. A* dung `f(n)=g(n)+h(n)` va heuristic Manhattan de di vong ngan hon thay vi duyet mu.

### Ngay 3 - Khu vuon cay phia dong - Local Search

Nhieu cay co muc do kho/heo khac nhau. Robot dung cac bien the local search nhu local beam, hill climbing va simulated annealing de toi uu thu tu xu ly toan bo cay con lai, roi thuc hien muc tieu dau tien trong ke hoach tot nhat.

### Ngay 4 - Khu bai dat suong mu - Online Search

Robot chi thay ban kinh 3 o. Mot so o sut/ngap bi an duoi suong va chi bi phat hien khi den gan. Khi gap block moi, robot cap nhat ban do va re-plan.

### Ngay 5 - Khu quy hoach trung tam - CSP Backtracking

Ong Minh muon trong lai theo rang buoc: corn va tomato khong duoc nam ke nhau. Robot dung backtracking de gan crop cho tung o truoc khi gieo.

### Ngay 6 - Khu hang cay can bao ve - Minimax

Mot ke pha hoai tranh den cac o cay de lam hong cay vua trong. AGRI-1 dung minimax/alpha-beta de chon o can bao ve, gia su doi thu cung phan ung toi uu.

## File chinh

- `code/algorithms.py`: cac thuat toan rieng: BFS, A*, Local Search, Online vision update, CSP Backtracking, Minimax.
- `code/ai_controller.py`: dieu phoi robot, animation, panel thong tin va visualization.
- `code/level.py`: cau hinh 6 khu moi, spawn, farm tiles, obstacles, hidden blocks va enemy spawn.
- `code/main.py`: vong lap Pygame va phim 1-6 de doi khu.
