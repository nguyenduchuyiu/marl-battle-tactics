import math
import random
import json
import os
import shutil # For copying model files
from config import Config

config = Config()

DEFAULT_ELO = config.DEFAULT_ELO  # Elo khởi điểm cho agent mới
K_FACTOR = config.ELO_K_FACTOR       # Hệ số K, ảnh hưởng đến mức độ thay đổi Elo sau mỗi trận

class EloOpponent:
    """Lớp đại diện cho một đối thủ trong pool với thông tin Elo."""
    def __init__(self, policy_path, elo=DEFAULT_ELO, games_played=0, name=None):
        self.policy_path = policy_path  # Đường dẫn đến file model (.pt) của agent
        self.elo = round(elo)
        self.games_played = games_played
        # Tên định danh, ví dụ: "agent_gen_1_ep500.pt"
        self.name = name if name else os.path.basename(policy_path)

    def to_dict(self):
        """Chuyển đối tượng thành dictionary để lưu vào JSON."""
        return {
            "policy_path": self.policy_path, # Sẽ lưu tên file thay vì đường dẫn tuyệt đối
            "elo": self.elo,
            "games_played": self.games_played,
            "name": self.name
        }

    @classmethod
    def from_dict(cls, data, model_dir):
        """Tạo đối tượng từ dictionary (đọc từ JSON)."""
        # policy_path trong JSON là tên file, cần ghép với model_dir
        full_policy_path = os.path.join(model_dir, os.path.basename(data["policy_path"]))
        return cls(
            policy_path=full_policy_path,
            elo=data.get("elo", DEFAULT_ELO),
            games_played=data.get("games_played", 0),
            name=data.get("name")
        )

def calculate_expected_score(elo_a, elo_b):
    """Tính điểm kỳ vọng của người chơi A khi đấu với người chơi B."""
    return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))

def update_elo_ratings(elo_a, elo_b, score_a, k_factor=K_FACTOR):
    """
    Cập nhật điểm Elo cho hai người chơi.
    score_a: 1 nếu A thắng, 0 nếu A thua (B thắng), 0.5 nếu hòa.
    Trả về: (new_elo_a, new_elo_b)
    """
    expected_a = calculate_expected_score(elo_a, elo_b)
    # score_b là kết quả của B, ví dụ: nếu A thắng (score_a=1) thì B thua (score_b=0)
    score_b = 1 - score_a

    new_elo_a = elo_a + k_factor * (score_a - expected_a)
    # expected_b có thể tính là 1 - expected_a
    expected_b = 1 - expected_a
    new_elo_b = elo_b + k_factor * (score_b - expected_b)

    return round(new_elo_a), round(new_elo_b)


class EloManager:
    """Quản lý pool các đối thủ và điểm Elo của chúng."""
    def __init__(self, pool_registry_file="elo_opponent_pool.json",
                 max_pool_size=20,
                 opponent_models_storage_dir="elo_opponent_models"):
        self.pool_registry_file = pool_registry_file
        self.max_pool_size = max_pool_size
        # Thư mục lưu trữ các file model .pt của các agent trong pool
        self.opponent_models_storage_dir = opponent_models_storage_dir
        os.makedirs(self.opponent_models_storage_dir, exist_ok=True)

        self.opponent_pool = []  # Danh sách các đối tượng EloOpponent
        self.load_pool_registry()

    def save_pool_registry(self):
        """Lưu thông tin metadata của pool (không phải file model) vào JSON."""
        # Chỉ lưu tên file model, không lưu đường dẫn tuyệt đối vào JSON
        pool_data = []
        for opponent in self.opponent_pool:
            data = opponent.to_dict()
            data["policy_path"] = os.path.basename(opponent.policy_path) # Chỉ lưu tên file
            pool_data.append(data)

        try:
            with open(self.pool_registry_file, 'w') as f:
                json.dump(pool_data, f, indent=4)
            # print(f"Elo opponent pool registry saved to {self.pool_registry_file}")
        except IOError as e:
            print(f"Error saving Elo pool registry: {e}")

    def load_pool_registry(self):
        """Tải thông tin metadata của pool từ JSON."""
        if not os.path.exists(self.pool_registry_file):
            print(f"Elo pool registry file {self.pool_registry_file} not found. Starting with an empty pool.")
            return

        try:
            with open(self.pool_registry_file, 'r') as f:
                pool_data = json.load(f)
                self.opponent_pool = [EloOpponent.from_dict(data, self.opponent_models_storage_dir) for data in pool_data]
            print(f"Elo opponent pool registry loaded. Pool size: {len(self.opponent_pool)}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading Elo pool registry: {e}. Starting with an empty pool.")
            self.opponent_pool = []

    def add_opponent_to_pool(self, source_policy_path, initial_elo=DEFAULT_ELO, name_prefix="agent_snapshot"):
        """
        Thêm một đối thủ mới vào pool.
        Sao chép file model của agent vào thư mục lưu trữ của pool.
        source_policy_path: Đường dẫn đến file .pt của agent cần thêm.
        """
        if not os.path.exists(source_policy_path):
            print(f"Error: Source policy file {source_policy_path} not found. Cannot add to pool.")
            return None

        # Tạo tên file duy nhất trong thư mục pool, ví dụ: agent_snapshot_ep1000_elo1200.pt
        base_name = os.path.basename(source_policy_path)
        # Tránh ghi đè nếu file đã tồn tại bằng cách thêm một số thứ tự hoặc timestamp
        # new_opponent_filename = f"{name_prefix}_{base_name}" # Đơn giản hóa
        # Để đảm bảo duy nhất, có thể dùng timestamp hoặc uuid nếu cần
        timestamp = str(int(random.random()*1e9)) # Cách đơn giản để tạo sự khác biệt
        new_opponent_filename = f"{name_prefix}_{timestamp}_{base_name}"
        destination_policy_path = os.path.join(self.opponent_models_storage_dir, new_opponent_filename)

        try:
            shutil.copyfile(source_policy_path, destination_policy_path)
            print(f"Copied policy from {source_policy_path} to {destination_policy_path}")
        except Exception as e:
            print(f"Error copying policy file {source_policy_path} to pool's storage: {e}")
            return None

        if len(self.opponent_pool) >= self.max_pool_size:
            # Nếu pool đầy, loại bỏ đối thủ có Elo thấp nhất (hoặc chiến lược khác)
            self.opponent_pool.sort(key=lambda op: op.elo) # Sắp xếp theo Elo tăng dần
            removed_opponent = self.opponent_pool.pop(0)
            print(f"Opponent pool full. Removed opponent: {removed_opponent.name} (Elo: {removed_opponent.elo})")
            try:
                if os.path.exists(removed_opponent.policy_path):
                    os.remove(removed_opponent.policy_path) # Xóa file model vật lý
                    print(f"Deleted model file from storage: {removed_opponent.policy_path}")
            except OSError as e:
                print(f"Error deleting model file {removed_opponent.policy_path}: {e}")

        new_opponent = EloOpponent(policy_path=destination_policy_path, elo=initial_elo, name=new_opponent_filename)
        self.opponent_pool.append(new_opponent)
        self.save_pool_registry()
        print(f"Added new opponent to pool: {new_opponent.name} (Elo: {new_opponent.elo})")
        return new_opponent

    def select_opponent(self, current_agent_elo=None, strategy="random_weighted"):
        """
        Chọn một đối thủ từ pool.
        Chiến lược:
        - "random": chọn ngẫu nhiên.
        - "random_weighted": chọn ngẫu nhiên, ưu tiên các agent có Elo gần với current_agent_elo.
        - "elo_closest": chọn agent có Elo gần nhất.
        - "strongest": chọn agent có Elo cao nhất.
        - "weakest": chọn agent có Elo thấp nhất.
        """
        if not self.opponent_pool:
            print("Warning: Opponent pool is empty.")
            return None

        if strategy == "random":
            return random.choice(self.opponent_pool)
        elif strategy == "strongest":
            return max(self.opponent_pool, key=lambda op: op.elo)
        elif strategy == "weakest":
            return min(self.opponent_pool, key=lambda op: op.elo)
        elif strategy == "elo_closest" and current_agent_elo is not None:
            return min(self.opponent_pool, key=lambda op: abs(op.elo - current_agent_elo))
        elif strategy == "random_weighted" and current_agent_elo is not None:
            # Tính trọng số dựa trên sự khác biệt Elo, ưu tiên những agent có Elo gần
            # Agent có Elo càng gần thì xác suất được chọn càng cao
            weights = []
            for op in self.opponent_pool:
                # Tránh chia cho 0, sự khác biệt nhỏ -> trọng số lớn
                diff = abs(op.elo - current_agent_elo)
                # Trọng số tỷ lệ nghịch với bình phương sự khác biệt Elo (cộng 1 để tránh chia cho 0)
                # Có thể thử các hàm trọng số khác nhau
                weight = 1 / (diff**2 + 1)
                weights.append(weight)
            
            if sum(weights) == 0: # Nếu tất cả trọng số là 0 (hiếm)
                return random.choice(self.opponent_pool)
            return random.choices(self.opponent_pool, weights=weights, k=1)[0]
        else: # Mặc định là random nếu chiến lược không hợp lệ hoặc thiếu thông tin
            return random.choice(self.opponent_pool)

    def update_opponent_stats(self, opponent_name_or_path, new_elo, games_increment=1):
        """Cập nhật Elo và số trận đã chơi của một đối thủ cụ thể trong pool."""
        found = False
        for opponent in self.opponent_pool:
            # Tìm theo tên hoặc đường dẫn đầy đủ
            if opponent.name == opponent_name_or_path or \
               os.path.abspath(opponent.policy_path) == os.path.abspath(opponent_name_or_path):
                opponent.elo = round(new_elo)
                opponent.games_played += games_increment
                found = True
                print(f"Updated stats for opponent: New Elo = {opponent.elo}, Games Played = {opponent.games_played}")
                break
        if not found:
            print(f"Warning: Opponent '{opponent_name_or_path}' not found in pool for stats update.")
        self.save_pool_registry() # Lưu thay đổi

    def get_opponent_by_name(self, name):
        """Lấy đối tượng EloOpponent từ pool bằng tên."""
        for opponent in self.opponent_pool:
            if opponent.name == name:
                return opponent
        return None

    def __len__(self):
        return len(self.opponent_pool)

# --- Ví dụ sử dụng (có thể đặt trong if __name__ == '__main__': để test) ---
if __name__ == '__main__':
    # Tạo thư mục và file model giả để test
    test_source_model_dir = "temp_source_models_for_elo_test"
    os.makedirs(test_source_model_dir, exist_ok=True)
    dummy_model_path1 = os.path.join(test_source_model_dir, "test_agent_v1.pt")
    dummy_model_path2 = os.path.join(test_source_model_dir, "test_agent_v2.pt")
    with open(dummy_model_path1, 'w') as f: f.write("dummy_model_data_v1")
    with open(dummy_model_path2, 'w') as f: f.write("dummy_model_data_v2")

    # Khởi tạo EloManager cho test
    test_registry_file = "test_elo_registry.json"
    test_models_storage = "test_elo_models_storage"

    # Xóa file và thư mục test cũ nếu có
    if os.path.exists(test_registry_file): os.remove(test_registry_file)
    if os.path.exists(test_models_storage): shutil.rmtree(test_models_storage)

    elo_mgr = EloManager(
        pool_registry_file=test_registry_file,
        max_pool_size=3,
        opponent_models_storage_dir=test_models_storage
    )

    # Thêm đối thủ
    op1_obj = elo_mgr.add_opponent_to_pool(dummy_model_path1, initial_elo=1200, name_prefix="test_v1")
    op2_obj = elo_mgr.add_opponent_to_pool(dummy_model_path2, initial_elo=1250, name_prefix="test_v2")

    if op1_obj and op2_obj:
        print(f"Pool size: {len(elo_mgr)}")
        print(f"Opponent 1: {op1_obj.name}, Elo: {op1_obj.elo}")
        print(f"Opponent 2: {op2_obj.name}, Elo: {op2_obj.elo}")

        # Giả sử agent hiện tại (Blue) có Elo 1220 và thắng op1_obj
        current_blue_elo = 1220
        opponent_to_play_with = elo_mgr.get_opponent_by_name(op1_obj.name)

        if opponent_to_play_with:
            print(f"\nBlue (Elo {current_blue_elo}) is playing against {opponent_to_play_with.name} (Elo {opponent_to_play_with.elo})")
            score_for_blue = 1.0 # Blue thắng
            new_blue_elo, new_opponent_elo = update_elo_ratings(current_blue_elo, opponent_to_play_with.elo, score_for_blue)

            print(f"Match result: Blue wins.")
            print(f"  Blue Elo: {current_blue_elo} -> {new_blue_elo}")
            print(f"  Opponent {opponent_to_play_with.name} Elo: {opponent_to_play_with.elo} -> {new_opponent_elo}")

            # Cập nhật Elo cho agent hiện tại và đối thủ trong pool
            current_blue_elo = new_blue_elo
            elo_mgr.update_opponent_stats(opponent_to_play_with.name, new_opponent_elo)

        # Chọn đối thủ tiếp theo
        next_opponent = elo_mgr.select_opponent(current_agent_elo=current_blue_elo, strategy="random_weighted")
        if next_opponent:
            print(f"\nNext opponent selected (random_weighted): {next_opponent.name} (Elo: {next_opponent.elo})")

        # Thêm một agent nữa để test max_pool_size
        dummy_model_path3 = os.path.join(test_source_model_dir, "test_agent_v3.pt")
        with open(dummy_model_path3, 'w') as f: f.write("dummy_model_data_v3")
        op3_obj = elo_mgr.add_opponent_to_pool(dummy_model_path3, initial_elo=1100, name_prefix="test_v3") # Elo thấp
        op4_obj = elo_mgr.add_opponent_to_pool(dummy_model_path3, initial_elo=1300, name_prefix="test_v4") # Sẽ đẩy op3_obj ra nếu max_pool_size=3

    # Dọn dẹp file test
    if os.path.exists(test_source_model_dir): shutil.rmtree(test_source_model_dir)
    if os.path.exists(test_registry_file): os.remove(test_registry_file)
    if os.path.exists(test_models_storage): shutil.rmtree(test_models_storage)
    print("\nTest cleanup complete.")
