def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# farm tiles
farm_tiles = []
for x in range(29, 34):
    for y in range(22, 26 + 1):
        farm_tiles.append((x, y))

spawn_tile = (34, 24)

# Con đường zigzag mong muốn
path = [
    (33, 24), (33, 25), (33, 26),
    (32, 26), (32, 25), (32, 24), (32, 23), (32, 22),
    (31, 22), (31, 23), (31, 24), (31, 25), (31, 26),
    (30, 26), (30, 25)
]
path_set = set(path)

# Hàm tính score cố định
def get_score(tile, dry):
    return dry * 1.4 - 4 * heuristic(spawn_tile, tile)

# Chúng ta cần tìm bộ dryness cho 25 ô sao cho Hill Climbing đi đúng path
# Chúng ta có thể dùng thuật toán leo núi hoặc backtracking để tìm bộ dryness này.
# Thử dùng backtracking đơn giản để gán dryness từ 10 đến 95 cho các ô.
# Để tăng tốc, chúng ta có thể đặt một số ràng buộc:
# - Các ô OFF-PATH có dryness cố định là 10.
# - Các ô trên path có dryness tăng dần dọc theo path? Không nhất thiết, miễn là score tăng dần dọc theo path.
# - dryness của mỗi ô trên path nằm trong khoảng [15, 95].

import random

def solve():
    dryness = {tile: 10 for tile in farm_tiles}
    
    # Ràng buộc cơ bản:
    # Tại mỗi bước k từ 0 đến len(path)-2:
    # Đứng tại path[k], láng giềng kề chưa cứu (trong remaining) có score cao nhất phải là path[k+1],
    # và score của path[k+1] phải lớn hơn score của path[k].
    # Sau bước cuối cùng path[-1], không có láng giềng kề chưa cứu nào có score lớn hơn path[-1].
    
    # Để đơn giản, hãy gán dryness sao cho score tăng dần dọc theo path:
    # Score(path[k+1]) > Score(path[k])
    # Tức là: dryness(path[k+1])*1.4 - 4*dist(path[k+1]) > dryness(path[k])*1.4 - 4*dist(path[k])
    # Từ đó: dryness(path[k+1]) > dryness(path[k]) + 2.85 * (dist(path[k+1]) - dist(path[k]))
    
    # Đồng thời, tại mỗi bước k, với mỗi láng giềng kề 'nbr' của path[k] chưa cứu và nbr != path[k+1]:
    # Ta phải có Score(path[k+1]) > Score(nbr)
    # Tức là: dryness(path[k+1])*1.4 - 4*dist(path[k+1]) > dryness(nbr)*1.4 - 4*dist(nbr)
    # Nếu nbr là ô OFF-PATH (dryness = 10), điều này rất dễ thỏa mãn vì dryness(nbr)=10 rất nhỏ.
    # Nhưng nếu nbr là một ô khác trên path (ví dụ ô path[j] với j > k+1):
    # Ta phải có: dryness(path[k+1])*1.4 - 4*dist(path[k+1]) > dryness(path[j])*1.4 - 4*dist(path[j])
    
    # Hãy thử chạy một thuật toán tối ưu hóa (random search) để tìm bộ dryness thỏa mãn:
    for attempt in range(100000):
        # Khởi tạo ngẫu nhiên dryness cho các ô trên path sao cho dryness tăng dần dọc theo path
        current_dryness = {tile: 10 for tile in farm_tiles}
        val = 20
        for tile in path:
            val += random.randint(2, 6)
            current_dryness[tile] = min(95, val)
            
        # Kiểm tra xem bộ dryness này có chạy đúng path không
        remaining = set(farm_tiles)
        current = spawn_tile
        steps = []
        
        # Bước đầu tiên từ spawn_tile
        x, y = current
        candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        candidates = [t for t in candidates if t in remaining]
        if candidates:
            # Chọn ô lân cận gần spawn nhất
            current = min(candidates, key=lambda t: heuristic(spawn_tile, t))
            steps.append(current)
            remaining.remove(current)
            
        success = True
        for target in path:
            if current != target:
                success = False
                break
                
            # Tìm bước tiếp theo
            x, y = current
            candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            neighbors = [t for t in candidates if t in remaining]
            
            current_score = get_score(current, current_dryness[current])
            better_neighbors = [t for t in neighbors if get_score(t, current_dryness[t]) > current_score]
            
            if not better_neighbors:
                # Nếu đây là ô cuối cùng của path, thì dừng là đúng
                if current == path[-1]:
                    break
                else:
                    success = False
                    break
            
            next_tile = max(better_neighbors, key=lambda t: (get_score(t, current_dryness[t]), -heuristic(current, t), t))
            
            # Kiểm tra xem next_tile có khớp với ô tiếp theo trên path không
            idx = path.index(current)
            if idx + 1 < len(path) and next_tile == path[idx + 1]:
                current = next_tile
                remaining.remove(current)
                steps.append(current)
            else:
                success = False
                break
                
        if success and len(steps) == len(path):
            print(f"Found solution at attempt {attempt}!")
            print("Dryness values:")
            for y in range(22, 27):
                row_str = ""
                for x in range(29, 34):
                    row_str += f"{current_dryness[(x, y)]:3d} "
                print(row_str)
            return current_dryness
            
    print("Failed to find a solution using random search.")
    return None

solve()
