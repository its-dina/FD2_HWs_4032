import numpy as np
from math import isclose
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    raise ImportError("Please install SciPy library: pip install scipy")

def angular_velocity_to_euler(w_b):

    wx, wy, wz = w_b
    

    phi, theta, psi = 0.0, 0.0, 0.0
    
    T = np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])
    

    euler_rates = np.dot(T, np.array([wx, wy, wz]))
    phi_dot, theta_dot, psi_dot = euler_rates
    
    return {
        'angles_Euler': [phi, theta, psi],
        'rate_angles_Euler': [phi_dot, theta_dot, psi_dot]
    }

def transform_matrix_info(mat, tol=1e-3):

    mat = np.array(mat, dtype=float)
    

    if mat.shape != (3, 3):
        return "Input matrix does not meet rotation matrix conditions"
    

    det_val = np.linalg.det(mat)
    if not isclose(det_val, 1.0, abs_tol=tol):
        return "Input matrix does not meet rotation matrix conditions"
    

    should_be_identity = np.dot(mat.T, mat)
    I = np.eye(3)
    if not np.allclose(should_be_identity, I, atol=tol):
        return "Input matrix does not meet rotation matrix conditions"
    

    rot = R.from_matrix(mat)
    
  
    quaternion = rot.as_quat()
    

    rotation_vector = rot.as_rotvec()
    

    euler_angles = rot.as_euler('zyx', degrees=True)
    
    return {
        "Euler_angles": euler_angles.tolist(),
        "Rotation_vector": rotation_vector.tolist(),
        "Quaternion_vector": quaternion.tolist()
    }

def calculate_angular_velocities(V, bank_angle_deg):

    g = 9.81  # m/s^2
    phi_rad = np.radians(bank_angle_deg)
    R = V**2 / (g * np.tan(phi_rad))
    omega_inertial = V / R
    angular_velocity_inertial = [0.0, 0.0, omega_inertial]
    angular_velocity_body = [
        0.0,
        omega_inertial * np.sin(phi_rad),
        omega_inertial * np.cos(phi_rad)
    ]
    return {
        "Angular_velocity_in_inertial_frame": angular_velocity_inertial,
        "Angular_velocity_in_body_frame": angular_velocity_body
    }


if __name__ == "__main__":
    # Question 1 
    w_b = [0.33, 0.28, 0.16]
    result1 = angular_velocity_to_euler(w_b)
    print("Question 1 - Euler Angles:", result1['angles_Euler'])
    print("Question 1 - Euler Angle Rates:", result1['rate_angles_Euler'])
    
    # Question 2
    C_t = [
        [0.2802, 0.1387, 0.9499],
        [0.1962, 0.9603, -0.1981],
        [-0.9397, 0.2418, 0.2418]
    ]
    result2 = transform_matrix_info(C_t, tol=1e-3)
    if isinstance(result2, str):
        print("Question 2:", result2)
    else:
        print("Question 2 - Matrix is valid.")
        print("Question 2 - Output:", result2)
    
    # Question 3
    V = 250.0
    bank_angle = 60.0
    result3 = calculate_angular_velocities(V, bank_angle)
    print("Question 3 - Inertial frame:", result3["Angular_velocity_in_inertial_frame"])
    print("Question 3 - Body frame:", result3["Angular_velocity_in_body_frame"])
