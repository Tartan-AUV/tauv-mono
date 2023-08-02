import rospy
import numpy as np
import cv_bridge
import sensor_msgs.msg
from typing import Optional
import threading
import math
import scipy as sp
import sys
import rawpy
import collections
from PIL import Image


class Debluer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lock.acquire()
        self.frame_id = rospy.get_param('~frame_id')
        self.cv_bridge = cv_bridge.CvBridge()
        self.img_sub = rospy.Subscriber(f'vehicle/{self.frame_id}/color/image_raw', Image, self.handle_img)
        self.depth_sub = rospy.Subscriber(f'vehicle/{self.frame_id}/depth/image_raw', Image, self.handle_depth)
        self.pub = rospy.Publisher(f'vision/{self.frame_id}/debluer', sensor_msgs.msg.Image)
        self.depth: Optional[sensor_msgs.msg.Image] = None
        self.lock.release()

    def start(self):
        rospy.spin()
    
    def handle_depth(self, msg):
        self.lock.acquire()
        self.depth = msg
        self.lock.release()

    def find_backscatter_estimation_points(self, img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
        z_max, z_min = np.max(depths), np.min(depths)
        min_depth = z_min + (min_depth_percent * (z_max - z_min))
        z_ranges = np.linspace(z_min, z_max, num_bins + 1)
        img_norms = np.mean(img, axis=2)
        points_r = []
        points_g = []
        points_b = []
        for i in range(len(z_ranges) - 1):
            a, b = z_ranges[i], z_ranges[i+1]
            locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
            norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
            arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
            points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
            points_r.extend([(z, p[0]) for n, p, z in points])
            points_g.extend([(z, p[1]) for n, p, z in points])
            points_b.extend([(z, p[2]) for n, p, z in points])
        return np.array(points_r), np.array(points_g), np.array(points_b)

    def find_backscatter_values(self, B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
        B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
        z_max, z_min = np.max(depths), np.min(depths)
        max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
        coefs = None
        best_loss = np.inf
        def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
            val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
            return val
        def loss(B_inf, beta_B, J_prime, beta_D_prime):
            val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
            return val
        bounds_lower = [0,0,0,0]
        bounds_upper = [1,5,1,5]
        for _ in range(restarts):
            try:
                optp, pcov = sp.optimize.curve_fit(
                    f=estimate,
                    xdata=B_depths,
                    ydata=B_vals,
                    p0=np.random.random(4) * bounds_upper,
                    bounds=(bounds_lower, bounds_upper),
                )
                l = loss(*optp)
                if l < best_loss:
                    best_loss = l
                    coefs = optp
            except RuntimeError as re:
                print(re, file=sys.stderr)
        if best_loss > max_mean_loss:
            print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
            BD = (slope * depths) + intercept
            return BD, np.array([slope, intercept])
        return estimate(depths, *coefs), coefs


    def estimate_illumination(self, img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
        D = img - B
        avg_cs = np.zeros_like(img)
        avg_cs_prime = np.copy(avg_cs)
        sizes = np.zeros(num_neighborhoods)
        locs_list = [None] * num_neighborhoods
        for label in range(1, num_neighborhoods + 1):
            locs_list[label - 1] = np.where(neighborhood_map == label)
            sizes[label - 1] = np.size(locs_list[label - 1][0])
        for _ in range(max_iters):
            for label in range(1, num_neighborhoods + 1):
                locs = locs_list[label - 1]
                size = sizes[label - 1] - 1
                avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
            new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
            if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
                break
            avg_cs = new_avg_cs
        return f * self.denoise_bilateral(np.maximum(0, avg_cs))

    def estimate_wideband_attentuation(self, depths, illum, radius = 6, max_val = 10.0):
        eps = 1E-8
        BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
        mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
        refined_attenuations = self.denoise_bilateral(self.closing(np.maximum(0, BD * mask), self.disk(radius)))
        return refined_attenuations, []

    def calculate_beta_D(self, depths, a, b, c, d):
        return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

    def filter_data(self, X, Y, radius_fraction=0.01):
        idxs = np.argsort(X)
        X_s = X[idxs]
        Y_s = Y[idxs]
        x_max, x_min = np.max(X), np.min(X)
        radius = (radius_fraction * (x_max - x_min))
        ds = np.cumsum(X_s - np.roll(X_s, (1,)))
        dX = [X_s[0]]
        dY = [Y_s[0]]
        tempX = []
        tempY = []
        pos = 0
        for i in range(1, ds.shape[0]):
            if ds[i] - ds[pos] >= radius:
                tempX.append(X_s[i])
                tempY.append(Y_s[i])
                idxs = np.argsort(tempY)
                med_idx = len(idxs) // 2
                dX.append(tempX[med_idx])
                dY.append(tempY[med_idx])
                pos = i
            else:
                tempX.append(X_s[i])
                tempY.append(Y_s[i])
        return np.array(dX), np.array(dY)

    def refine_wideband_attentuation(self, depths, illum, estimation, restarts=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
        eps = 1E-8
        z_max, z_min = np.max(depths), np.min(depths)
        min_depth = z_min + (min_depth_fraction * (z_max - z_min))
        max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
        coefs = None
        best_loss = np.inf
        locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
        def calculate_reconstructed_depths(depths, illum, a, b, c, d):
            eps = 1E-5
            res = -np.log(illum + eps) / (self.calculate_beta_D(depths, a, b, c, d) + eps)
            return res
        def loss(a, b, c, d):
            return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))
        dX, dY = self.filter_data(depths[locs], estimation[locs], radius_fraction)
        for _ in range(restarts):
            try:
                optp, pcov = sp.optimize.curve_fit(
                    f=self.calculate_beta_D,
                    xdata=dX,
                    ydata=dY,
                    p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                    bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
                L = loss(*optp)
                if L < best_loss:
                    best_loss = L
                    coefs = optp
            except RuntimeError as re:
                print(re, file=sys.stderr)
   
        if best_loss > max_mean_loss:
            print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs], estimation[locs])
            BD = (slope * depths + intercept)
            return l * BD, np.array([slope, intercept])
        print(f'Found best loss {best_loss}', flush=True)
        BD = l * self.calculate_beta_D(depths, *coefs)
        return BD, coefs

    def recover_image(self, img, depths, B, beta_D, nmap):
        res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
        res = np.maximum(0.0, np.minimum(1.0, res))
        res[nmap == 0] = 0
        res = self.scale(self.wbalance_no_red_10p(res))
        res[nmap == 0] = img[nmap == 0]
        return res

    def recover_image_S4(self, img, B, illum, nmap):
        eps = 1E-8
        res = (img - B) / (illum + eps)
        res = np.maximum(0.0, np.minimum(1.0, res))
        res[nmap == 0] = img[nmap == 0]
        return self.scale(self.wbalance_no_red_gw(res))

    def construct_neighborhood_map(self, depths, epsilon=0.05):
        eps = (np.max(depths) - np.min(depths)) * epsilon
        nmap = np.zeros_like(depths).astype(np.int32)
        n_neighborhoods = 1
        while np.any(nmap == 0):
            locs_x, locs_y = np.where(nmap == 0)
            start_index = np.random.randint(0, len(locs_x))
            start_x, start_y = locs_x[start_index], locs_y[start_index]
            q = collections.deque()
            q.append((start_x, start_y))
            while not len(q) == 0:
                x, y = q.pop()
                if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                    nmap[x, y] = n_neighborhoods
                    if 0 <= x < depths.shape[0] - 1:
                        x2, y2 = x + 1, y
                        if nmap[x2, y2] == 0:
                            q.append((x2, y2))
                    if 1 <= x < depths.shape[0]:
                        x2, y2 = x - 1, y
                        if nmap[x2, y2] == 0:
                            q.append((x2, y2))
                    if 0 <= y < depths.shape[1] - 1:
                        x2, y2 = x, y + 1
                        if nmap[x2, y2] == 0:
                            q.append((x2, y2))
                    if 1 <= y < depths.shape[1]:
                        x2, y2 = x, y - 1
                        if nmap[x2, y2] == 0:
                            q.append((x2, y2))
            n_neighborhoods += 1
        zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
        if len(zeros_size_arr) > 0:
            nmap[nmap == zeros_size_arr[0][0]] = 0 #reset largest background to 0
        return nmap, n_neighborhoods - 1

    def find_closest_label(self, nmap, start_x, start_y):
        mask = np.zeros_like(nmap).astype(np.bool)
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
                if nmap[x, y] != 0:
                    return nmap[x, y]
                mask[x, y] = True
                if 0 <= x < nmap.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if not mask[x2, y2]:
                        q.append((x2, y2))
                if 1 <= x < nmap.shape[0]:
                    x2, y2 = x - 1, y
                    if not mask[x2, y2]:
                        q.append((x2, y2))
                if 0 <= y < nmap.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if not mask[x2, y2]:
                        q.append((x2, y2))
                if 1 <= y < nmap.shape[1]:
                    x2, y2 = x, y - 1
                    if not mask[x2, y2]:
                        q.append((x2, y2))

    def refine_neighborhood_map(self, nmap, min_size = 10, radius = 3):
        refined_nmap = np.zeros_like(nmap)
        vals, counts = np.unique(nmap, return_counts=True)
        neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
        num_labels = 1
        for label, size in neighborhood_sizes:
            if size >= min_size and label != 0:
                refined_nmap[nmap == label] = num_labels
                num_labels += 1
        for label, size in neighborhood_sizes:
            if size < min_size and label != 0:
                for x, y in zip(*np.where(nmap == label)):
                    refined_nmap[x, y] = self.find_closest_label(refined_nmap, x, y)
        refined_nmap = self.closing(refined_nmap, self.square(radius))
        return refined_nmap, num_labels - 1


    def load_image_and_depth_map(self, img_fname, depths_fname, size_limit = 1024):
        depths = Image.open(depths_fname)
        img = Image.fromarray(rawpy.imread(img_fname).postprocess())
        img.thumbnail((size_limit, size_limit), Image.ANTIALIAS)
        depths = depths.resize(img.size, Image.ANTIALIAS)
        return np.float32(img) / 255.0, np.array(depths)

    def wbalance_gw(self, img):
        dr = 1.0 / np.mean(img[:, :, 0])
        dg = 1.0 / np.mean(img[:, :, 1])
        db = 1.0 / np.mean(img[:, :, 2])
        dsum = dr + dg + db
        dr = dr / dsum * 3.
        dg = dg / dsum * 3.
        db = db / dsum * 3.

        img[:, :, 0] *= dr
        img[:, :, 1] *= dg
        img[:, :, 2] *= db
        return img

    def wbalance_10p(self, img):
        dr = 1.0 / np.mean(np.sort(img[:, :, 0], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
        dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
        db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
        dsum = dr + dg + db
        dr = dr / dsum * 3.
        dg = dg / dsum * 3.
        db = db / dsum * 3.

        img[:, :, 0] *= dr
        img[:, :, 1] *= dg
        img[:, :, 2] *= db
        return img

    def wbalance_no_red_10p(self, img):
        dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
        db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
        dsum = dg + db
        dg = dg / dsum * 2.
        db = db / dsum * 2.
        img[:, :, 0] *= (db + dg) / 2
        img[:, :, 1] *= dg
        img[:, :, 2] *= db
        return img

    def wbalance_no_red_gw(self, img):
        dg = 1.0 / np.mean(img[:, :, 1])
        db = 1.0 / np.mean(img[:, :, 2])
        dsum = dg + db
        dg = dg / dsum * 2.
        db = db / dsum * 2.

        img[:, :, 0] *= (db + dg) / 2
        img[:, :, 1] *= dg
        img[:, :, 2] *= db
        return img

    def scale(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def run_pipeline(self, img, depths, args):
        print('Estimating backscatter...', flush=True)
        ptsR, ptsG, ptsB = self.find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=args.min_depth)

        print('Finding backscatter coefficients...', flush=True)
        Br, coefsR = self.find_backscatter_values(ptsR, depths, restarts=25)
        Bg, coefsG = self.find_backscatter_values(ptsG, depths, restarts=25)
        Bb, coefsB = self.find_backscatter_values(ptsB, depths, restarts=25)

        print('Constructing neighborhood map...', flush=True)
        nmap, _ = self.construct_neighborhood_map(depths, 0.1)

        print('Refining neighborhood map...', flush=True)
        nmap, n = self.refine_neighborhood_map(nmap, 50)

        print('Estimating illumination...', flush=True)
        illR = self.estimate_illumination(img[:, :, 0], Br, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
        illG = self.estimate_illumination(img[:, :, 1], Bg, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
        illB = self.estimate_illumination(img[:, :, 2], Bb, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
        ill = np.stack([illR, illG, illB], axis=2)

        print('Estimating wideband attenuation...', flush=True)
        beta_D_r, _ = self.estimate_wideband_attentuation(depths, illR)
        refined_beta_D_r, coefsR = self.refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=args.spread_data_fraction, l=args.l)
        beta_D_g, _ = self.estimate_wideband_attentuation(depths, illG)
        refined_beta_D_g, coefsG = self.refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=args.spread_data_fraction, l=args.l)
        beta_D_b, _ = self.estimate_wideband_attentuation(depths, illB)
        refined_beta_D_b, coefsB = self.refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=args.spread_data_fraction, l=args.l)

        print('Reconstructing image...', flush=True)
        B = np.stack([Br, Bg, Bb], axis=2)
        beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
        recovered = self.recover_image(img, depths, B, beta_D, nmap)

        return recovered

    def handle_img(self, msg):
        self.lock.acquire()
        if self.depth is None:
            self.lock.release()
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        depth_img = self.cv_bridge.imgmsg_to_cv2(self.depth, desired_encoding='mono16')
        result = self.run_pipeline(img, depth_img)
        result_msg = self.cv_bridge.cv2_to_imgmsg(result, encoding='bgr8')
        self.pub.publish(result_msg)
        self.lock.release()
    
    def main():
        rospy.init_node('debluer')
        n = Debluer()
        n.start()