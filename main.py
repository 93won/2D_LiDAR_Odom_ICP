from utils import *
import matplotlib.pyplot as plt
from icp import *


if __name__ == "__main__":
    lidar, lidar_data, name = ReadData(0)

    start = 0
    end = 5000
    gap = 5

    pose = [0, 0, 0]

    traj = []

    for i in range(start, end, gap):

        scan_before_local = lidar.ReadAScan(lidar_data, i, 60) # (N , 2)
        scan_current_local = lidar.ReadAScan(lidar_data, i+gap, 60) # (N' ,2)

        scan_before_global = localToGlobal(pose, scan_before_local)
        scan_current_global = localToGlobal(pose, scan_current_local)

        plt.clf()
        plt.xlim([-50, 50])
        plt.ylim([-50, 50])

        plt.plot(pose[0], pose[1], 'o', color='blue', markersize=3)
        traj.append(pose)

        traj_array = np.array(traj)
        plt.plot(traj_array[:, 0], traj_array[:, 1], color='black')

        T = icp(scan_current_global, scan_before_global)

        pose_T = v2t(pose)
        pose = t2v(np.copy(np.dot(T, pose_T)))

        frame = np.ones((3, scan_current_global.shape[0]))
        frame[:2, :] = scan_current_global.T
        result = (T @ frame)[:2, :].T

        plt.plot(result[:, 0], result[:, 1], 'o', markersize=1, color='red')
        plt.savefig('result/{}.png'.format(i), dpi=300)