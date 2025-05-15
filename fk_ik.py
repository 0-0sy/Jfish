import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from vispy import scene
from vispy.scene import visuals


## ALL UNITS ARE in mm

# Link lengths and offsets
a1, a2, a3, a4, a5 = 130., 115., 110., 24., 104.

# Denavit-Hartenberg Parameters
DH = np.array([[ a1, np.pi/2,   0,   0],
               [ a2, 0,         0,   0],
               [ a3, 0,         0,   0],
               [ a4, -np.pi/2,  0,   0],
               [ a5, 0,         0,   0]], dtype=np.float64)

# Generate Transformation Matrix
def gen_T(a, alpha, d, theta):
    cos_t, sin_t, cos_a, sin_a = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[cos_t, -sin_t * cos_a,  sin_t * sin_a,  a * cos_t],
                     [sin_t,  cos_t * cos_a, -cos_t * sin_a,  a * sin_t],
                     [0,      sin_a,          cos_a,          d        ],
                     [0,      0,              0,              1        ]])

# Forward Kinematics
def FK(theta):
    T01 = gen_T(DH[0, 0], DH[0, 1], DH[0, 2], DH[0, 3] + theta[0])
    T12 = gen_T(DH[1, 0], DH[1, 1], DH[1, 2], DH[1, 3] + theta[1])
    T23 = gen_T(DH[2, 0], DH[2, 1], DH[2, 2], DH[2, 3] + theta[2])
    T34 = gen_T(DH[3, 0], DH[3, 1], DH[3, 2], DH[3, 3] + theta[3])
    T45 = gen_T(DH[4, 0], DH[4, 1], DH[4, 2], DH[4, 3] + theta[4])

    T02 = np.matmul(T01, T12)
    T03 = np.matmul(T02, T23)
    T04 = np.matmul(T03, T34)
    T05 = np.matmul(T04, T45)

    return [T01, T02, T03, T04, T05]

# Inverse Kinematics
def IK(l, d, alpha, heading):
    p05 = np.array([l * np.sin(alpha), -l * np.cos(alpha), d])
    heading = heading / np.linalg.norm(heading)
    p04 = p05 - a5 * heading
    p03 = p04 - a4 * heading

    th1 = np.arctan2(p03[1], p03[0])

    p01 = np.array([a1 * np.cos(th1), a1 * np.sin(th1), 0])
    pw = p03 - p01
    r = np.linalg.norm(pw[:2])
    s = pw[2]

    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    D = np.clip(D, -1.0, 1.0)
    th3 = np.arccos(D)

    phi1 = np.arctan2(s, r)
    phi2 = np.arctan2(a3 * np.sin(th3), a2 + a3 * np.cos(th3))
    th2 = phi1 - phi2

    # 회전 행렬 구성
    R0_1 = gen_T(a1, np.pi/2, 0, th1)[:3, :3]
    R1_2 = gen_T(a2, 0, 0, th2)[:3, :3]
    R2_3 = gen_T(a3, 0, 0, th3)[:3, :3]
    R0_3 = R0_1 @ R1_2 @ R2_3

    # θ4
    x3 = R0_3 @ np.array([1, 0, 0])
    x4_desired = (p04 - p03) / np.linalg.norm(p04 - p03)
    th4 = np.arccos(np.clip(np.dot(x3, x4_desired), -1.0, 1.0))

    # θ5
    R3_4 = gen_T(a4, -np.pi/2, 0, th4)[:3, :3]
    R0_4 = R0_3 @ R3_4
    x4 = R0_4 @ np.array([1, 0, 0])
    cross = np.cross(x4, heading)
    sign = np.sign(np.dot(R0_4[:, 2], cross))
    dot = np.clip(np.dot(x4, heading), -1.0, 1.0)
    th5 = sign * np.arccos(dot)

    return [th1, th2, th3, th4, th5]



# PyQt5 GUI with Vispy visualization
class RobotVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theta_list = [0] * 5
        self.joint_positions = [np.array([0, 0, 0]) for _ in range(6)]

        # Create main widget and layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Create Vispy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1200, 800))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.scale_factor = 400
        self.layout.addWidget(self.canvas.native)
        self.view.camera.center = (-200, 0, 0)

        # create Grid
        grid = visuals.GridLines(color=(0.8, 0.8, 0.8, 1.0))
        grid.transform = scene.transforms.STTransform(scale=(1000, 1000, 1))
        self.view.add(grid)

        # Initialize Vispy visuals
        self.markers = visuals.Markers()
        self.arrows = []  # X, Y, Z 축 화살표 추가
        self.lines = visuals.Line(color='blue', width=4)
        self.view.add(self.markers)
        self.view.add(self.lines)

        # 화살표를 X, Y, Z 축당 생성
        for _ in range(18):  # 총 18개 화살표 (5 조인트 × 3 축 + EE 3축)
            arrow = visuals.Arrow(width=2)
            self.view.add(arrow)
            self.arrows.append(arrow)

        # 원점좌표계 생성
        self.arrows[15].set_data(pos=np.array([(0,0,0), (20,0,0)]), color='red')
        self.arrows[16].set_data(pos=np.array([(0,0,0), (0,20,0)]), color='green')
        self.arrows[17].set_data(pos=np.array([(0,0,0), (0,0,20)]), color='blue')

        # Initialize mesh for end effector
        vertices, faces = self.create_cylinder_mesh(radius=197, height=12)
        self.end_effector_mesh = visuals.Mesh(vertices=vertices, faces=faces, color=(1, 0, 0, 0.5))
        self.view.add(self.end_effector_mesh)

        # Add sliders
        for i, initial_value in enumerate([0, -45, 90, 45, 0]):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-115)
            slider.setMaximum(115)
            slider.setValue(initial_value)  # 초기 슬라이더 위치 설정
            slider.valueChanged.connect(lambda value, i=i: self.slider_moved(value, i))
            self.layout.addWidget(slider)
            self.slider_moved(slider.value(), i)

        self.update_FK()

    def slider_moved(self, value, slider_number):
        self.theta_list[slider_number] = np.radians(value)
        self.update_FK()

        # IK의 입력값은 계산된 FK의 EE pos + orientation에서.
        l = np.sqrt(self.joint_positions[5][0]**2+self.joint_positions[5][1]**2)
        d = self.joint_positions[5][2]
        alpha = np.arctan2(self.joint_positions[5][1], self.joint_positions[5][0]) + np.pi/2
        angle = self.EE_orientation
        # print(f"l={l:.1f}, d={d:.1f}, a={alpha:.3f}")
        # print(f"x: {angle[0]:.2f}, y: {angle[1]:.2f}, z: {angle[2]:.2f}")

        # IK의 최종 정답은 FK의 입력값에서.
        real_J1 = self.theta_list[0]
        real_J2 = self.theta_list[1]
        real_J3 = self.theta_list[2]
        real_J4 = self.theta_list[3]
        real_J5 = self.theta_list[4]
        print(f"input:[{real_J1:.2f}, {real_J2:.2f}, {real_J3:.2f}, {real_J4:.2f}, {real_J5:.2f}]")

        # solve IK
        J = IK(l, d, alpha, angle)
        print(f"output:[{J[0]:.2f}, {J[1]:.2f}, {J[2]:.2f}, {J[3]:.2f}, {J[4]:.2f}]")


    def create_cylinder_mesh(self, radius, height, segments=36):
        """위아래 면이 없는 원기둥 메쉬 데이터를 생성"""
        angle_step = 2 * np.pi / segments
        vertices = []
        faces = []

        # 원기둥의 측면 정점 생성
        for i in range(segments):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # 상단 원점 (z = height/2)
            vertices.append([x, y, height / 2])

            # 하단 원점 (z = -height/2)
            vertices.append([x, y, -height / 2])

        # 측면 면 정의
        for i in range(segments):
            top1 = i * 2
            bottom1 = top1 + 1
            top2 = (i * 2 + 2) % (segments * 2)
            bottom2 = (i * 2 + 3) % (segments * 2)

            # 측면 면 추가
            faces.append([top1, bottom1, top2])
            faces.append([top2, bottom1, bottom2])

        # y축으로 기본 90도 회전 적용
        vertices = np.array(vertices)
        rotation_matrix = np.array([
            [0,  0, 1],
            [0,  1, 0],
            [-1, 0, 0]
        ])
        rotated_vertices = vertices @ rotation_matrix.T

        return rotated_vertices, np.array(faces)

    def update_FK(self):
        Ts = FK(self.theta_list)
        self.joint_positions = [np.array([0, 0, 0])] + [T[:3, 3] for T in Ts]

        # Update lines (links)
        self.lines.set_data(pos=np.array(self.joint_positions), color='white')

        # Update arrows (axes)
        for i in range(5):
            pos = self.joint_positions[i+1]

            # 각 축의 방향 벡터 계산
            x_dir = Ts[i][:3, 0] * 20  # X축 방향 벡터 (빨강)
            y_dir = Ts[i][:3, 1] * 20  # Y축 방향 벡터 (초록)
            z_dir = Ts[i][:3, 2] * 20  # Z축 방향 벡터 (파랑)

            # X축 화살표
            self.arrows[i * 3].set_data(pos=np.array([pos, pos + x_dir]), color='red')
            # Y축 화살표
            self.arrows[i * 3 + 1].set_data(pos=np.array([pos, pos + y_dir]), color='green')
            # Z축 화살표
            self.arrows[i * 3 + 2].set_data(pos=np.array([pos, pos + z_dir]), color='blue')

        # Update markers (joints)
        colors = ['blue'] * 5 + ['red']
        
        self.markers.set_data(
            np.array(self.joint_positions),
            face_color=colors,
            size=12
        )

        # 엔드이펙터 위치와 회전
        end_effector_pos = self.joint_positions[-1]  # 엔드이펙터 위치
        end_effector_orientation = Ts[-1][:3, :3]   # 엔드이펙터 회전 행렬 (3x3)

        # 원기둥 데이터 생성
        vertices, faces = self.create_cylinder_mesh(radius=197, height=12)

        # 회전과 이동 적용
        rotated_vertices = (end_effector_orientation @ vertices.T).T  # 회전 적용
        translated_vertices = rotated_vertices + end_effector_pos  # 위치 적용

        # 메쉬 업데이트
        self.end_effector_mesh.set_data(vertices=translated_vertices, faces=faces)

        # save EE orientation for IK
        self.EE_orientation = Ts[4][:3, 0]

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotVisualizer()
    window.show()
    sys.exit(app.exec_())