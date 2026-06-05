# Pydew Valley - One Map AI A* Demo

Bản này được chỉnh trực tiếp từ code/map gốc của Pydew Valley.

## Đã giữ lại
- Map gốc `data/map.tmx`
- Asset gốc trong `graphics/`, `audio/`, `font/`
- Logic gốc: `Level`, `Player`, `SoilLayer`, `Generic`, `Tree`, `Water`, camera, overlay
- Cơ chế cuốc đất, gieo hạt, tưới nước của `SoilLayer`

## Đã tối giản/thêm
- Game tự start, không cần bấm Start.
- Tắt input bàn phím khi AI chạy.
- Thêm `code/ai_controller.py`.
- AI dùng A* để tìm đường đến các ô đất gần player.
- AI tự thao tác: cuốc đất -> gieo hạt -> tưới nước.
- Có panel hiển thị trạng thái thuật toán trên màn hình.
- `main.py` tự đổi working directory về thư mục `code` để không lỗi đường dẫn asset.

## Cách chạy

```powershell
cd "D:\24133056\Ai\pydew_ai_original_map\code"
python -m pip install pygame pytmx
python main.py
```

Hoặc chạy từ thư mục khác cũng được vì `main.py` đã tự `chdir`.

```powershell
python "D:\24133056\Ai\pydew_ai_original_map\code\main.py"
```

## File chỉnh chính
- `code/main.py`
- `code/level.py`
- `code/player.py`
- `code/ai_controller.py`


## Bản chỉnh mới

- Spawn nhân vật ở khu đất phía góc phải map.
- Đào sẵn một khu đất 5x5 tại tile `(41, 14)` đến `(45, 18)`.
- AI A* xử lý tối đa 25 ô đất demo.
