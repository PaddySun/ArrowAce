# 同于计算的快捷工具
import numpy as np

class MathTool:
    def __init__(self):
        pass

    def calculate_angle(self, point_a, point_b, point_c):
        """
        计算由三个点构成的夹角。
        :param point_a: 第一个点的坐标 (x, y)
        :param point_b: 第二个点的坐标，作为角的顶点 (x, y)
        :param point_c: 第三个点的坐标 (x, y)
        :return: 夹角度数
        """
        vector_ab = np.array(point_a) - np.array(point_b)
        vector_bc = np.array(point_c) - np.array(point_b)
        dot_product = np.dot(vector_ab, vector_bc)
        magnitude_ab = np.linalg.norm(vector_ab)
        magnitude_bc = np.linalg.norm(vector_bc)
        angle_rad = np.arccos(dot_product / (magnitude_ab * magnitude_bc))
        return np.degrees(angle_rad)

    def midpoint(self, point_a, point_b):
        """
        计算两点之间的中点坐标。
        
        :param point_a: 第一个点的坐标 (x, y)
        :param point_b: 第二个点的坐标 (x, y)
        :return: 中点坐标
        """
        return ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)


    def check_body_alignment(self, left_shoulder, right_shoulder, hip_center):
        """
        检查身体侧倾角度是否接近理想值。
        
        :param left_shoulder: 左肩点坐标
        :param right_shoulder: 右肩点坐标
        :param hip_center: 髋关节中心点坐标，用于确定身体正面方向
        :return: 与垂直线的偏离角度
        """
        body_direction = right_shoulder - left_shoulder
        ideal_direction = np.array([1, 0])  # 假设面向目标为正东，简化处理
        return self.calculate_angle(hip_center, left_shoulder, right_shoulder) - self.calculate_angle(hip_center, (0, 0), ideal_direction)

    # 更多函数可以按需添加，如计算重心、直线一致性检查等