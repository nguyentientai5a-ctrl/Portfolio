"""
Mô phỏng Robot có Camera sử dụng Python
Ứng dụng: Điều khiển Robot + Xử lý ảnh kỹ thuật
Tác giả: Nguyễn Tiến Tài - Cơ điện tử - UET
"""

import numpy as np
import math

# ============================================================
# PHẦN 1: ĐIỀU KHIỂN ROBOT - Kinematics (Động học)
# ============================================================

class Robot2DOF:
    """
    Robot 2 bậc tự do (2-DOF) phẳng
    Mô phỏng cánh tay robot với 2 khớp quay
    """
    def __init__(self, L1=1.0, L2=0.8):
        self.L1 = L1   # Chiều dài link 1 (m)
        self.L2 = L2   # Chiều dài link 2 (m)
        self.theta1 = 0.0  # Góc khớp 1 (rad)
        self.theta2 = 0.0  # Góc khớp 2 (rad)

    def forward_kinematics(self, theta1, theta2):
        """
        Động học thuận: tính vị trí đầu robot từ góc khớp
        """
        self.theta1 = theta1
        self.theta2 = theta2

        # Vị trí khớp 2
        x1 = self.L1 * math.cos(theta1)
        y1 = self.L1 * math.sin(theta1)

        # Vị trí đầu cuối (end-effector)
        x2 = x1 + self.L2 * math.cos(theta1 + theta2)
        y2 = y1 + self.L2 * math.sin(theta1 + theta2)

        return (x1, y1), (x2, y2)

    def inverse_kinematics(self, x, y):
        """
        Động học ngược: tính góc khớp từ vị trí đầu robot mong muốn
        """
        # Kiểm tra điểm có trong vùng làm việc không
        dist = math.sqrt(x**2 + y**2)
        if dist > (self.L1 + self.L2) or dist < abs(self.L1 - self.L2):
            raise ValueError(f"Điểm ({x:.2f}, {y:.2f}) nằm ngoài vùng làm việc!")

        # Tính theta2 (cosine rule)
        cos_theta2 = (x**2 + y**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta2 = max(-1, min(1, cos_theta2))  # Clamp để tránh lỗi acos
        theta2 = math.acos(cos_theta2)

        # Tính theta1
        k1 = self.L1 + self.L2 * math.cos(theta2)
        k2 = self.L2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        return math.degrees(theta1), math.degrees(theta2)

    def workspace_radius(self):
        return self.L1 + self.L2


class PIDController:
    """
    Bộ điều khiển PID cho robot
    """
    def __init__(self, Kp=2.0, Ki=0.1, Kd=0.5, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, current):
        error = setpoint - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output, error

    def simulate_trajectory(self, target_angle, steps=100):
        """Mô phỏng quỹ đạo điều khiển đến góc mục tiêu"""
        current = 0.0
        trajectory = [current]
        errors = []

        for _ in range(steps):
            output, error = self.compute(target_angle, current)
            current += output * self.dt
            trajectory.append(current)
            errors.append(abs(error))

        return trajectory, errors


# ============================================================
# PHẦN 2: XỬ LÝ ẢNH - Computer Vision
# ============================================================

class ImageProcessor:
    """
    Xử lý ảnh kỹ thuật bằng NumPy (không cần OpenCV)
    Mô phỏng các thuật toán xử lý ảnh cơ bản
    """

    @staticmethod
    def create_test_image(size=64):
        """Tạo ảnh test với các hình dạng hình học"""
        img = np.zeros((size, size), dtype=np.uint8)
        # Vẽ hình chữ nhật (mô phỏng vật thể)
        img[10:30, 10:30] = 200   # Hình vuông trắng
        img[35:55, 35:55] = 150   # Hình vuông xám
        # Thêm nhiễu
        noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def grayscale_stats(image):
        """Phân tích thống kê ảnh grayscale"""
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "shape": image.shape
        }

    @staticmethod
    def threshold(image, thresh_val=128):
        """Phân ngưỡng ảnh (Binary Thresholding)"""
        binary = np.where(image > thresh_val, 255, 0).astype(np.uint8)
        white_pixels = int(np.sum(binary == 255))
        black_pixels = int(np.sum(binary == 0))
        return binary, white_pixels, black_pixels

    @staticmethod
    def apply_kernel(image, kernel):
        """
        Tích chập 2D (convolution) - nền tảng của xử lý ảnh
        Dùng cho: làm mờ, phát hiện cạnh, sharpening
        """
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh, j:j+kw].astype(np.float64)
                output[i, j] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_kernel(size=3, sigma=1.0):
        """Tạo kernel Gaussian để làm mờ ảnh"""
        k = size // 2
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x, y = i - k, j - k
                kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    @staticmethod
    def sobel_edge_detection(image):
        """Phát hiện cạnh bằng bộ lọc Sobel"""
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        h, w = image.shape
        padded = np.pad(image, 1, mode='edge').astype(np.float64)
        grad_x = np.zeros((h, w))
        grad_y = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                grad_x[i, j] = np.sum(region * Gx)
                grad_y[i, j] = np.sum(region * Gy)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def find_objects(binary_image, min_size=10):
        """
        Phát hiện vật thể trong ảnh nhị phân (Connected Components đơn giản)
        Trả về danh sách bounding boxes
        """
        objects = []
        visited = np.zeros_like(binary_image, dtype=bool)
        h, w = binary_image.shape

        for i in range(h):
            for j in range(w):
                if binary_image[i, j] > 0 and not visited[i, j]:
                    # BFS để tìm vùng liên thông
                    pixels = []
                    queue = [(i, j)]
                    while queue:
                        ci, cj = queue.pop(0)
                        if ci < 0 or ci >= h or cj < 0 or cj >= w:
                            continue
                        if visited[ci, cj] or binary_image[ci, cj] == 0:
                            continue
                        visited[ci, cj] = True
                        pixels.append((ci, cj))
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            queue.append((ci+di, cj+dj))

                    if len(pixels) >= min_size:
                        rows = [p[0] for p in pixels]
                        cols = [p[1] for p in pixels]
                        objects.append({
                            "bbox": (min(rows), min(cols), max(rows), max(cols)),
                            "area": len(pixels),
                            "centroid": (sum(rows)//len(rows), sum(cols)//len(cols))
                        })
        return objects


# ============================================================
# PHẦN 3: ROBOT CÓ CAMERA - Tích hợp
# ============================================================

class RobotWithCamera:
    """
    Robot tích hợp camera - kết hợp điều khiển và xử lý ảnh
    Mô phỏng hệ thống Pick-and-Place tự động
    """
    def __init__(self):
        self.robot = Robot2DOF(L1=1.0, L2=0.8)
        self.pid_joint1 = PIDController(Kp=3.0, Ki=0.05, Kd=0.8)
        self.pid_joint2 = PIDController(Kp=2.5, Ki=0.05, Kd=0.6)
        self.camera = ImageProcessor()
        self.current_pos = (0.0, 0.0)
        self.log = []

    def capture_and_detect(self):
        """Chụp ảnh và phát hiện vật thể"""
        img = self.camera.create_test_image(64)
        binary, white, black = self.camera.threshold(img, 100)
        objects = self.camera.find_objects(binary, min_size=5)
        self.log.append(f"[CAMERA] Phát hiện {len(objects)} vật thể")
        return objects, img

    def plan_path(self, objects):
        """Lập kế hoạch di chuyển đến vật thể"""
        if not objects:
            self.log.append("[ROBOT] Không có vật thể để gắp")
            return []

        waypoints = []
        for obj in objects:
            cy, cx = obj["centroid"]
            # Chuyển pixel → tọa độ thực (tỷ lệ 0.02 m/pixel)
            real_x = (cx - 32) * 0.02
            real_y = (cy - 32) * 0.02
            max_r = self.robot.workspace_radius()
            # Clamp vào vùng làm việc
            dist = math.sqrt(real_x**2 + real_y**2)
            if dist > max_r * 0.9:
                scale = max_r * 0.9 / dist
                real_x *= scale
                real_y *= scale
            waypoints.append((real_x, real_y))
            self.log.append(f"[PATH] Waypoint: ({real_x:.3f}, {real_y:.3f}) m")
        return waypoints

    def move_to(self, x, y):
        """Di chuyển robot đến tọa độ (x, y)"""
        try:
            t1_deg, t2_deg = self.robot.inverse_kinematics(x, y)
            traj1, err1 = self.pid_joint1.simulate_trajectory(t1_deg, steps=50)
            traj2, err2 = self.pid_joint2.simulate_trajectory(t2_deg, steps=50)
            joint1, joint2 = self.robot.forward_kinematics(
                math.radians(traj1[-1]), math.radians(traj2[-1])
            )
            self.current_pos = joint2
            final_err = math.sqrt(
                (joint2[0]-x)**2 + (joint2[1]-y)**2
            )
            self.log.append(
                f"[MOVE] Đến ({x:.3f}, {y:.3f}) | "
                f"Khớp: θ1={t1_deg:.1f}°, θ2={t2_deg:.1f}° | "
                f"Sai số: {final_err:.4f} m"
            )
            return True, t1_deg, t2_deg
        except ValueError as e:
            self.log.append(f"[LỖI] {e}")
            return False, 0, 0

    def run_pick_and_place(self):
        """Thực hiện quy trình Pick-and-Place đầy đủ"""
        print("=" * 60)
        print("  MÔ PHỎNG ROBOT PICK-AND-PLACE VỚI CAMERA")
        print("  Nguyễn Tiến Tài | Cơ điện tử | UET")
        print("=" * 60)

        # 1. Khởi tạo
        print("\n[1] Khởi tạo hệ thống...")
        print(f"    Robot 2-DOF: L1={self.robot.L1}m, L2={self.robot.L2}m")
        print(f"    Vùng làm việc: R = {self.robot.workspace_radius():.2f} m")

        # 2. Chụp ảnh & phát hiện
        print("\n[2] Camera quét môi trường...")
        objects, img = self.capture_and_detect()
        stats = self.camera.grayscale_stats(img)
        print(f"    Ảnh: {stats['shape']} | Mean={stats['mean']:.1f} | Std={stats['std']:.1f}")
        print(f"    Phát hiện: {len(objects)} vật thể")
        for i, obj in enumerate(objects):
            print(f"    Vật {i+1}: bbox={obj['bbox']}, area={obj['area']}px, centroid={obj['centroid']}")

        # 3. Xử lý ảnh
        print("\n[3] Xử lý ảnh...")
        blur_kernel = self.camera.gaussian_kernel(3, 1.0)
        blurred = self.camera.apply_kernel(img, blur_kernel)
        edges = self.camera.sobel_edge_detection(blurred)
        edge_density = float(np.mean(edges > 50))
        print(f"    Gaussian blur: kernel 3x3, σ=1.0")
        print(f"    Sobel edge: mật độ cạnh = {edge_density*100:.1f}%")

        # 4. Lập kế hoạch
        print("\n[4] Lập kế hoạch quỹ đạo...")
        waypoints = self.plan_path(objects)
        print(f"    {len(waypoints)} waypoint được tạo")

        # 5. Thực thi
        print("\n[5] Thực thi điều khiển robot...")
        success_count = 0
        for i, (wx, wy) in enumerate(waypoints):
            ok, t1, t2 = self.move_to(wx, wy)
            status = "✓ THÀNH CÔNG" if ok else "✗ THẤT BẠI"
            print(f"    Waypoint {i+1}: ({wx:.3f}, {wy:.3f}) → {status}")
            if ok:
                success_count += 1
                print(f"       θ1={t1:.1f}°, θ2={t2:.1f}°")

        # 6. Kết quả
        print("\n[6] KẾT QUẢ:")
        print(f"    Hoàn thành: {success_count}/{len(waypoints)} waypoint")
        print(f"    Tỷ lệ thành công: {success_count/max(len(waypoints),1)*100:.0f}%")
        print(f"    Vị trí cuối: ({self.current_pos[0]:.3f}, {self.current_pos[1]:.3f}) m")

        # 7. Demo IK/FK
        print("\n[7] Demo Kinematics:")
        test_pts = [(0.8, 0.5), (1.0, 0.3), (0.5, 1.0)]
        for px, py in test_pts:
            try:
                t1d, t2d = self.robot.inverse_kinematics(px, py)
                j1, j2 = self.robot.forward_kinematics(
                    math.radians(t1d), math.radians(t2d)
                )
                err = math.sqrt((j2[0]-px)**2 + (j2[1]-py)**2)
                print(f"    IK({px},{py}) → θ1={t1d:.1f}°, θ2={t2d:.1f}° | FK err={err:.6f}m")
            except ValueError as e:
                print(f"    IK({px},{py}) → {e}")

        # 8. Demo PID
        print("\n[8] Demo PID Controller (θ_target = 90°):")
        pid_demo = PIDController(Kp=2.0, Ki=0.1, Kd=0.5)
        traj, errs = pid_demo.simulate_trajectory(90.0, steps=80)
        settle_idx = next((i for i, e in enumerate(errs) if e < 1.0), len(errs))
        print(f"    Góc cuối: {traj[-1]:.2f}° (mục tiêu: 90°)")
        print(f"    Sai số cuối: {errs[-1]:.4f}°")
        print(f"    Thời gian ổn định (<1°): bước {settle_idx}/{len(errs)}")

        print("\n" + "=" * 60)
        print("  MÔ PHỎNG HOÀN TẤT")
        print("=" * 60)


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    # Đặt seed để kết quả tái lập được
    np.random.seed(42)

    system = RobotWithCamera()
    system.run_pick_and_place()
