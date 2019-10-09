import numpy as np
import os
import json
import heapq
from scipy.spatial import ConvexHull
import time
from sklearn.neighbors import KDTree

def my_print(fp, s):
    print(s)
    fp.write(s+'\n')

class symmetry():
    def __init__(self, type_, order, p=None, v=None):
        """
        type 0: translational;  1: refectinal;  2: rotational
        """
        global NOBBS
        assert type_ in [0, 1, 2]
        self.type = type_
        if v is None:
            assert p is None
            self.vector = np.zeros(3)
            self.point = np.zeros(3)
        else:
            self.vector = v / np.linalg.norm(v)
            self.point = p
        self.order = [x+NOBBS for x in order]
        self.times = len(order)
    
    def output(self, fp):
        global OBJS
        fp.write('%d\n' % self.times)
        for i in self.order:
            fp.write('%d %g %g %g\n' % (i, *OBJS[i]['obb'][:3]))
        fp.write('%d\n' % (self.type+1))
        fp.write('%g %g %g\n%g %g %g\n' % (*self.point, *self.vector))

    def point_reflect(self, x):
        return 2*np.dot(self.point-x, self.vector) * self.vector + x

    def axis_reflect(self, ax, h):
        hax = h*ax
        reflect_hax = np.dot(-hax, self.vector)*self.vector + hax*0.5
        ref_h = np.sqrt(np.sum(reflect_hax**2))
        ref_ax = reflect_hax / ref_h
        return ref_ax, ref_h*2

def load_obj(file_list):
    global PATH
    # global PMAX
    # global PMIN
    n = 0
    v_list = []
    f_list = []
    for fn in file_list:
        v = []
        f = []
        with open(os.path.join(PATH, 'objs', fn+'.obj'), 'r') as fp:
            for line in fp:
                l = line.split()
                if l[0] == 'v':
                    v.append([float(x) for x in l[1:]])
                elif l[0] == 'f':
                    f.append([int(x) for x in l[1:]])
                else:
                    my_print(log_fp, 'unknow obj %s' % l[0])
            v = np.asarray(v)
            f = np.asarray(f)
            u = np.unique(f)
            v = v[u-1]
            d = {x: i for i, x in enumerate(u)}
            for i in range(f.shape[0]):
                f[i, 0] = d[f[i, 0]] + 1
                f[i, 1] = d[f[i, 1]] + 1
                f[i, 2] = d[f[i, 2]] + 1
            f += n
        v_list.append(v)
        f_list.append(f)
        n += len(v)
    points = np.concatenate(v_list, 0)
    tri = np.concatenate(f_list, 0)
    # _min = np.min(points, 0)
    # _max = np.max(points, 0)
    # PMAX = np.max(np.stack([_max, PMAX], 0), 0)
    # PMIN = np.min(np.stack([_min, PMIN], 0), 0)
    return points, tri

def rotationArc(v0, v1):
    cross = np.cross(v0, v1)
    d = np.dot(v0, v1)
    s = np.sqrt((1+d) * 2)
    if np.abs(s) <= 1e-15:
        return -1, None
    cross = cross / s
    return 0, np.concatenate([cross, [s*0.5]])

def quatToMatrix(quat):
    xx = quat[0]*quat[0]
    yy = quat[1]*quat[1]
    zz = quat[2]*quat[2]
    xy = quat[0]*quat[1]
    xz = quat[0]*quat[2]
    yz = quat[1]*quat[2]
    wx = quat[3]*quat[0]
    wy = quat[3]*quat[1]
    wz = quat[3]*quat[2]

    matrix = np.eye(4)
    matrix[0, 0] = 1 - 2 * ( yy + zz )
    matrix[1, 0] =     2 * ( xy - wz )
    matrix[2, 0] =     2 * ( xz + wy )

    matrix[0, 1] =     2 * ( xy + wz )
    matrix[1, 1] = 1 - 2 * ( xx + zz )
    matrix[2, 1] =     2 * ( yz - wx )

    matrix[0, 2] =     2 * ( xz - wy )
    matrix[1, 2] =     2 * ( yz + wx )
    matrix[2, 2] = 1 - 2 * ( xx + yy )

    # matrix[3, 0] = matrix[3, 1] = matrix[3, 2] = 0.0
    # matrix[0, 3] = matrix[1, 3] = matrix[2, 3] = 0.0
    # matrix[3, 3] = 1.0
    return matrix

def planeToMatrix(plane):
    ref = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    for i in range(3):
        check, quat = rotationArc(ref[i], plane[:3])
        if check == 0:
            break
    matrix = quatToMatrix(quat)
    matrix[3, 0] = matrix[1, 0] * -plane[3] + matrix[3, 0]
    matrix[3, 1] = matrix[1, 1] * -plane[3] + matrix[3, 1]
    matrix[3, 2] = matrix[1, 2] * -plane[3] + matrix[3, 2]
    return matrix

def computeBestFitPlane(verts):
    vcount = verts.shape[0]
    kOrigin = np.mean(verts, 0)
    kDiff = verts - kOrigin
    C = np.dot(kDiff.T, kDiff) / vcount
    _, ev = np.linalg.eigh(C)
    plane = np.zeros(4)
    kNormal = ev[:, 0]
    ax = np.argmax(np.abs(kNormal))
    if kNormal[ax] < 0:
        kNormal = -kNormal
    plane[0] = kNormal[0]
    plane[1] = kNormal[1]
    plane[2] = kNormal[2]
    plane[3] = -np.dot(kNormal, kOrigin)
    return plane

def eulerToQuat(roll, pitch, yaw):
    roll  *= 0.5
    pitch *= 0.5
    yaw   *= 0.5
    cr = np.cos(roll)
    cp = np.cos(pitch)
    cy = np.cos(yaw)
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sy = np.sin(yaw)
    cpcy = cp * cy
    spsy = sp * sy
    spcy = sp * cy
    cpsy = cp * sy
    quat = np.zeros(4)
    quat[0] = sr * cpcy - cr * spsy
    quat[1] = cr * spcy + sr * cpsy
    quat[2] = cr * cpsy - sr * spcy
    quat[3] = cr * cpcy + sr * spsy
    return quat

def computeOBB(verts, matrix):
    p = verts - matrix[3, :3]
    p = np.dot(p, matrix[:3, :3].T)
    bmin = np.min(p, 0)
    bmax = np.max(p, 0)
    sides = bmax - bmin
    center = (bmax + bmin) * 0.5
    matrix[3, 0] += matrix[0, 0] * center[0] + matrix[1, 0] * center[1] + matrix[2, 0] * center[2]
    matrix[3, 1] += matrix[0, 1] * center[0] + matrix[1, 1] * center[1] + matrix[2, 1] * center[2]
    matrix[3, 2] += matrix[0, 2] * center[0] + matrix[1, 2] * center[1] + matrix[2, 2] * center[2]
    return sides

def FitObb(verts):
    hull = ConvexHull(verts)
    verts = verts[hull.vertices]
    # compute AABB
    p_min = np.min(verts, 0)
    p_max = np.max(verts, 0)
    scale = p_max - p_min
    avolume = scale[0] * scale[1] * scale[2]
    # compute best fit plane
    plane = computeBestFitPlane(verts)
    # convert a plane equation to a 4x4 rotation matrix
    matrix = planeToMatrix(plane)
    # computeOBB
    sides = computeOBB(verts, matrix)
    volume = sides[0]*sides[1]*sides[2]
    # rotation
    stepSize = 3 # FS_SLOW_FIT
    FM_DEG_TO_RAD = ((2.0 * np.pi) / 360.0)
    refmatrix = matrix.copy()
    for a in range(0, 180, stepSize):
        quat = eulerToQuat(0, a*FM_DEG_TO_RAD, 0)
        matrix_tmp = quatToMatrix(quat)
        pmatrix = np.dot(matrix_tmp, refmatrix)
        psides = computeOBB(verts, pmatrix)
        v = psides[0]*psides[1]*psides[2]
        if v < volume:
            volume = v
            sides = psides.copy()
            matrix = pmatrix.copy()
    if avolume < volume:
        matrix = np.eye(4)
        matrix[3, 0] = (p_max[0]+p_min[0]) * 0.5
        matrix[3, 1] = (p_max[1]+p_min[1]) * 0.5
        matrix[3, 2] = (p_max[2]+p_min[2]) * 0.5
        sides = scale
    Axis0 = matrix[0, :3]
    Axisl = matrix[1, :3]
    Axis2 = matrix[2, :3]
    center = matrix[3, :3]
    return np.concatenate([center, Axis0, Axisl, Axis2, sides], 0)

def add_obj(d, o, s):
    v, f = load_obj(o['objs'])
    info = {'objs': o['objs'], 'v': v, 'f': f}
    if s not in d.keys():
        d[s] = [info]
    else:
        d[s].append(info)

def get_objs(o, d, s):
    s.append(o['name'])
    if 'children' not in o.keys() or 'caster' in o['name']:
        add_obj(d, o, ' '.join(s))
    else:
        for l in o['children']:
            get_objs(l, d, s)
    s.pop()

def getInterval(obb, axis):
    def projectPoint(p, axis):
        dot = axis.dot(p)
        return dot * np.linalg.norm(p, 2)
    centroid = obb[:3]
    l1 = obb[3:6] * obb[-3] * 0.5
    l2 = obb[6:9] * obb[-2] * 0.5
    l3 = obb[9:12] * obb[-1] * 0.5
    min_ = None
    max_ = None
    for i in range(2):
        for j in range(2):
            for k in range(2):
                point = centroid + l1 * (i * 2 - 1) + \
                    l2 * (j * 2 - 1) + l3 * (k * 2 - 1)
                v = projectPoint(point, axis)
                if not min_ and not max_:
                    min_ = v
                    max_ = v
                else:
                    min_ = min(min_, v)
                    max_ = max(max_, v)
    return min_, max_


def collision_detection(a, b):
    axis_a = np.split(a[3:-3], 3)
    axis_b = np.split(b[3:-3], 3)
    for i in range(3):
        min1, max1 = getInterval(a, axis_a[i])
        min2, max2 = getInterval(b, axis_a[i])
        if max1 < min2 or max2 < min1:
            return False
    for i in range(3):
        min1, max1 = getInterval(a, axis_b[i])
        min2, max2 = getInterval(b, axis_b[i])
        if max1 < min2 or max2 < min1:
            return False
    for i in range(3):
        for j in range(3):
            axis = np.cross(axis_a[i], axis_b[j])
            min1, max1 = getInterval(a, axis)
            min2, max2 = getInterval(b, axis)
            if max1 < min2 or max2 < min1:
                return False
    return True


def adj_detcet(objs):
    l = []
    n_obbs = len(objs)
    for i in range(n_obbs):
        for j in range(i + 1, n_obbs):
            if collision_detection(objs[i]['obb'], objs[j]['obb']):
                l.append((i, j))
    return l

def save_obb(fn, objs, adjs, syms, labels, save_obj=0):
    """
    save_obj: 0: not save;  1: save as one obj file; 2: save as independent objs
    """
    template1 = ' '.join(['%g']*15)+'\n'
    assert len(objs) == len(labels)
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'w') as fp:
        fp.write('N %d\n' % len(objs))
        for o in objs:
            fp.write(template1 % tuple(o['obb']))
        fp.write('C %d\n' % len(adjs))
        for a in adjs:
            fp.write('%d %d\n' % a)
        fp.write('S %d\n' % len(syms))
        for s in syms:
            s.output(fp)
        fp.write('L %d\n' % len(labels))
        fp.write('\n'.join(['%d']*len(labels)) % tuple(labels))

    if save_obj == 1:
        n = 0
        with open(fn[:-4] + '.obj', 'w') as fp:
            for i in range(len(objs)):
                fp.write('g %d\n' % i)
                np.savetxt(fp, objs[i]['v'], 'v %g %g %g')
                f = objs[i]['f'] + n
                n += objs[i]['v'].shape[0]
                np.savetxt(fp, f, 'f %d %d %d')
    elif save_obj == 2:
        for i in range(len(objs)):
            with open(fn[:-3] + '_%d.obj' % i, 'w') as fp:
                np.savetxt(fp, objs[i]['v'], 'v %g %g %g')
                np.savetxt(fp, objs[i]['f'], 'f %d %d %d')

def save_obj(fn, objs):
    n = 0
    with open(fn, 'w') as fp:
        for i in range(len(objs)):
            fp.write('g %d\n' % i)
            np.savetxt(fp, objs[i]['v'], 'v %g %g %g')
            f = objs[i]['f'] + n
            n += objs[i]['v'].shape[0]
            np.savetxt(fp, f, 'f %d %d %d')


def check_obb_same(obb1, obb2, flag=True):
    if flag:
        c1 = obb1[:3]
        c2 = obb2[:3]
        if np.linalg.norm(c1 - c2) > 0.1*np.min(obb1[-3:]):
            return False
    ax = obb1[3:6]
    a2xs = [obb2[3:6], obb2[6:9], obb2[9:12]]
    hs = list(obb2[-3:])
    max_i = 0
    max_k = -1
    for i in range(len(a2xs)):
        if np.abs(np.dot(ax, a2xs[i])) > max_k:
            max_k = np.abs(np.dot(ax, a2xs[i]))
            max_i = i
    if max_k < 0.95 or abs(obb1[-3] - hs[max_i]) > 0.5 * hs[max_i]:
        # print(max_k, abs(obb1[-3] - hs[max_i])/hs[max_i])
        return False
    a2xs.pop(max_i)
    hs.pop(max_i)
    ax = obb1[6:9]
    max_i = 0
    max_k = -1
    for i in range(len(a2xs)):
        if np.abs(np.dot(ax, a2xs[i])) > max_k:
            max_k = np.abs(np.dot(ax, a2xs[i]))
            max_i = i
    if max_k < 0.95 or abs(obb1[-2] - hs[max_i]) > 0.5 * hs[max_i]:
        # print(max_k, abs(obb1[-2] - hs[max_i])/hs[max_i])
        return False
    a2xs.pop(max_i)
    hs.pop(max_i)
    ax = obb1[9:12]
    max_i = 0
    max_k = -1
    for i in range(len(a2xs)):
        if np.abs(np.dot(ax, a2xs[i])) > max_k:
            max_k = np.abs(np.dot(ax, a2xs[i]))
            max_i = i
    if max_k < 0.95 or abs(obb1[-1] - hs[max_i]) > 0.5 * hs[max_i]:
        # print(max_k, abs(obb1[-1] - hs[max_i])/hs[max_i])
        return False
    return True

def translation(obbs, idx=None):
    if idx:
        obbs = [obbs[i] for i in idx]
    assert len(obbs) >= 2
    for i in range(1, len(obbs)):
        if not check_obb_same(obbs[0], obbs[i], False):
            return False, None
    c_list = [o[:3] for o in obbs]
    n_obbs = len(c_list)
    TM = []
    IDL = []
    TL = []
    n = 0
    t0 = None
    for i in range(n_obbs):
        for j in range(i+1, n_obbs):
            t = c_list[i] - c_list[j]
            if t0 is None:
                t0 = t
            if t0.dot(t) > 0:
                IDL.append([i, j])
            else:
                IDL.append([j, i])
            TM.append([np.linalg.norm(t), n])
            TL.append(t)
            n += 1

    TM = heapq.nsmallest(n_obbs - 1, TM, key=lambda x: x[0])
    dT = 0.1*TM[0][0]
    for i in range(0, n_obbs - 1):
        for j in range(i+1, n_obbs - 1):
            mi = TM[i][0]
            mj = TM[j][0]
            if abs(mi - mj) > dT:
                return False, None
            ti = TL[TM[i][1]]
            tj = TL[TM[j][1]]
            d = ti.dot(tj)/(mi*mj)
            if abs(d) < 0.97:
                return False, None

    TmpL = [*IDL[TM[0][1]]]
    while len(TmpL) < n_obbs:
        for i in range(1, n_obbs - 1):
            if IDL[TM[i][1]][0] == TmpL[-1]:
                TmpL.append(IDL[TM[i][1]][1])
            if IDL[TM[i][1]][1] == TmpL[0]:
                TmpL.insert(0, IDL[TM[i][1]][0])

    if idx:
        order = [idx[i] for i in TmpL]
    else:
        order = TmpL
    return True, symmetry(0, order)

def rotation(obbs):
    n_obbs = len(obbs)
    c_list = [o[:3] for o in obbs]
    assert n_obbs >= 3
    TM = []
    IDL = []
    n = 0
    for i in range(n_obbs):
        for j in range(i+1, n_obbs):
            t = c_list[i] - c_list[j]
            IDL.append([i, j])
            TM.append([np.linalg.norm(t), n])
            n += 1
    TM = heapq.nsmallest(n_obbs, TM, key=lambda x: x[0])
    dT = 0.2*TM[0][0]
    for i in range(n_obbs):
        for j in range(i+1, n_obbs):
            mi = TM[i][0]
            mj = TM[j][0]
            if abs(mi-mj) > dT:
                return False, None
    
    NB = [[] for i in range(n_obbs)]
    for i in range(n_obbs):
        NB[IDL[TM[i][1]][0]].append(IDL[TM[i][1]][1])
        NB[IDL[TM[i][1]][1]].append(IDL[TM[i][1]][0])
    TmpL = [NB[0][0], 0, NB[0][1]]
    while len(TmpL) < n_obbs:
        sf = TmpL[1]
        if NB[TmpL[0]][0] != sf and NB[TmpL[0]][0] != TmpL[-1]:
            TmpL.insert(0, NB[TmpL[0]][0])
        elif NB[TmpL[0]][1] != sf and NB[TmpL[0]][1] != TmpL[-1]:
            TmpL.insert(0, NB[TmpL[0]][1])
        sl = TmpL[-2]
        if NB[TmpL[-1]][0] != sl and NB[TmpL[-1]][0] != TmpL[0]:
            TmpL.append(NB[TmpL[-1]][0])
        elif NB[TmpL[-1]][1] != sl and NB[TmpL[-1]][1] != TmpL[0]:
            TmpL.append(NB[TmpL[-1]][1])
    center = np.zeros(3)
    normal = np.zeros(3)
    for i in range(len(TmpL)):
        itn = i + 1
        if itn == len(TmpL):
            itn = 0
        itnn = itn + 1
        if itnn == len(TmpL):
            itnn = 0
        center += c_list[i]
        normal += np.cross(c_list[itn]-c_list[i], c_list[itnn]-c_list[itn])
    center /= n_obbs
    normal = normal / np.linalg.norm(normal)
    return True, symmetry(2, TmpL, center, normal)

def reflect(obbs, n1=0, n2=1):
    a = obbs[n1][:3]
    b = obbs[n2][:3]
    center_ = (a + b) * 0.5
    axis_ = b - a
    symm = symmetry(1, [n1, n2], center_, axis_)
    if np.max(np.abs(symm.vector)) < 0.97: #? check axis align?
        return False, None
    ref_c = symm.point_reflect(a)
    ref_ax1, ref_h1 = symm.axis_reflect(obbs[n1][3:6], obbs[n1][-3])
    ref_ax2, ref_h2 = symm.axis_reflect(obbs[n1][6:9], obbs[n1][-2])
    ref_ax3, ref_h3 = symm.axis_reflect(obbs[n1][9:12], obbs[n1][-1])
    ref_obb = np.concatenate([ref_c, ref_ax1, ref_ax2, ref_ax3, np.array([ref_h1, ref_h2, ref_h3])])
    if not check_obb_same(ref_obb, obbs[n2]):
        # print('ref fail')
        return False, None
    return True, symm


def find_reflections(obbs, axises=[0, 1, 2]):
    def find_reflect(l):
        for ax in axises: #? consider axis y?
            # axis = np.zeros(3)
            for i in range(len(l)):
                for j in range(i+1, len(l)):
                    tmpv = obbs[l[i]][:3] - obbs[l[j]][:3]
                    tmpv = tmpv / np.sqrt(np.sum(tmpv**2))
                    if abs(tmpv[ax]) > 0.97: #? chooce
                        check, ref = reflect(obbs, l[i], l[j])
                        if check:
                            return True, ref, i, j
        return False, None, -1, -1

    lis = [x for x in range(len(obbs))]
    ref_list = []
    while len(lis) > 1:
        check, r1, a, b = find_reflect(lis)
        if not check:
            return ref_list
        lis.pop(a)
        lis.pop(b-1)
        ref_list.append(r1)
    return ref_list

def merge(obj_list):
    v_list = []
    f_list = []
    n_verts = 0
    for k in obj_list:
        v_list.append(k['v'])
        f_list.append(k['f'] + n_verts)
        n_verts += k['v'].shape[0]
    v = np.concatenate(v_list, 0)
    f = np.concatenate(f_list, 0)
    obb = FitObb(v)
    return {'v': v, 'f': f, 'obb': obb}

def find_reflections_translation(obbs):
    d = np.zeros(3)
    for i in range(len(obbs)):
        for j in range(i+1, len(obbs)):
            d += np.abs(obbs[i][:3] - obbs[j][:3])
    ax = np.argmax(d)
    centers = [[x[:3], i] for i, x in enumerate(obbs)]
    centers.sort(key=lambda x: x[0][ax])
    def find_reflect(l):
        for i in range(len(l)):
            for j in range(len(l)-1, i, -1):
                tmpv = l[i][0] - l[j][0]
                tmpv = tmpv / np.sqrt(np.sum(tmpv**2))
                if abs(tmpv[ax]) > 0.97: #? chooce
                    check, ref = reflect(obbs, l[i][1], l[j][1])
                    if check:
                        return True, ref, i, j
        return False, None, -1, -1
    ref_list = []
    while len(centers) > 1:
        check, r1, a, b = find_reflect(centers)
        if not check:
            return ref_list
        centers.pop(a)
        centers.pop(b-1)
        ref_list.append(r1)
    return ref_list

def IsInsideObb(obb, vp, p):
    dv = p - vp[6]
    d = obb[3:6].dot(dv)
    if d<0.0 or d>obb[-3]:
        return False
    d = obb[6:9].dot(dv)
    if d<0.0 or d>obb[-2]:
        return False
    d = obb[9:12].dot(dv)
    if d<0.0 or d>obb[-1]:
        return False
    return True

def adj_detcet_2(obbs, dNR=0.0):
    adjs = set()
    vp = [[] for i in range(len(obbs))]
    es = [[] for i in range(len(obbs))]
    m_dESR = 0.01
    m_e = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
    for i in range(len(obbs)):
        cent = obbs[i][:3]
        hsize0 = obbs[i][-3] * 0.5
        hsize1 = obbs[i][-2] * 0.5
        hsize2 = obbs[i][-1] * 0.5
        axis0 = obbs[i][3:6]
        axis1 = obbs[i][6:9]
        axis2 = obbs[i][9:12]
        vp[i].append(cent + axis0*hsize0 + axis1*hsize1 + axis2*hsize2)
        vp[i].append(cent + axis0*hsize0 - axis1*hsize1 + axis2*hsize2)
        vp[i].append(cent + axis0*hsize0 - axis1*hsize1 - axis2*hsize2)
        vp[i].append(cent + axis0*hsize0 + axis1*hsize1 - axis2*hsize2)
        vp[i].append(cent - axis0*hsize0 + axis1*hsize1 + axis2*hsize2)
        vp[i].append(cent - axis0*hsize0 - axis1*hsize1 + axis2*hsize2)
        vp[i].append(cent - axis0*hsize0 - axis1*hsize1 - axis2*hsize2)
        vp[i].append(cent - axis0*hsize0 + axis1*hsize1 - axis2*hsize2)
        for j in range(len(m_e)):
            esp = vp[i][m_e[j][0]]
            eep = vp[i][m_e[j][1]]
            ed = eep - esp
            l = np.linalg.norm(ed)
            N = int(l/m_dESR) + 1
            s = 1.0/(N-1)
            es[i].append(esp)
            for k in range(1, N-1):
                es[i].append(es[i][-1]+ed*s)
            es[i].append(eep)
    kdts = []
    for i in range(len(obbs)):
        kdts.append(KDTree(np.asarray(vp[i]), leaf_size=2))
    for i in range(len(obbs)):
        for j in range(len(obbs)):
            if i == j:
                continue
            bConn = False
            for s in range(len(es[i])):
                if IsInsideObb(obbs[j], vp[j], es[i][s]):
                    bConn = True
                else:
                    d, _ = kdts[j].query(np.expand_dims(es[i][s], 0))
                    if d*d < dNR:
                        bConn = True
                if bConn:
                    break
            if bConn:
                a = min(i, j)
                b = max(i, j)
                adjs.add((a, b))
    return sorted(list(adjs), key=lambda x:x[0])

def main():
    global PATH
    global NOBBS
    global OBJS
    # global PMAX
    # global PMIN
    with open(PATH + '/result_after_merging.json', 'r') as f:
        h = json.load(f)[0]
    instances = {}
    OBJS = []
    NOBBS = 0
    if len(h['children']) > 5:
        my_print(log_fp, 'Pass')
        return False
    for l in h['children']:
        if 'seat' in l['name']:
            add_obj(instances, l, 'chair %s' % l['name'])
        else:
            get_objs(l, instances, ['chair'])
    # points_center = (PMAX + PMIN) * 0.5
    # points_sacle = 1 / np.max(PMAX - PMIN)
    # rota_matrix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    labels = []
    OBJS = []
    syms = []
    NOBBS = 0
    success = True

    for k in instances.keys():
        #print(k, len(instances[k]))
        obb_l = []
        for j in instances[k]:
            # j['v'] = (j['v'] - points_center)
            # j['v'] = j['v']
            j['obb'] = FitObb(j['v'])
            obb_l.append(j['obb'])
        label_l = k.split(' ')
        # obb_l = [x['obb'] for x in instances[k]]
        n_obb = len(obb_l)
        if len(instances[k]) > 1:
            if label_l[1] == 'chair_head':
                pass
                # if n_obb == 1:
                #     pass
                # elif n_obb == 2:
                #     check, ref = reflect(obb_l)
                #     if check:
                #         syms.append(ref)
                #     else:
                #         my_print(log_fp, '%s not reflectional' % k)
                #         success = False
            elif label_l[1] == 'chair_seat':
                # check, symm = translation(obb_l)
                # if not check:
                    # my_print(log_fp, '%s not translational' % k)
                    # success = False
                    #break 
                # else:
                    # syms.append(symm)
                pass
            elif label_l[1] == 'chair_back':
                if label_l[2] == 'back_surface':
                    if n_obb == 1:
                        pass
                    elif n_obb == 2:
                        check, symm = reflect(obb_l)
                        if not check:
                            my_print(log_fp, '%s not reflectional' % k)
                            success = False
                            break
                        else:
                            syms.append(symm)
                    else:
                        check, symm = translation(obb_l)
                        if not check:
                            my_print(log_fp, '%s not translational' % k)
                            r_list = find_reflections_translation(obb_l)
                            if r_list:
                                syms.extend(r_list)
                            else:
                                my_print(log_fp, 'Back Surface find reflection Fail!')
                                success = False
                            # success = False
                                break
                        else:
                            syms.append(symm)
                elif label_l[2] == 'back_frame':
                    if n_obb == 1:
                        pass
                    elif n_obb == 2:
                        check, symm = reflect(obb_l)
                        if not check:
                            my_print(log_fp, '%s not reflectional' % k)
                            success = False
                            break
                        else:
                            syms.append(symm)
                    elif n_obb == 3:
                        my_print(log_fp, '3 back frame!')
                        ref_list = find_reflections(obb_l, [0, 2, 1])
                        if ref_list:
                            syms.extend(ref_list)
                        else:
                            my_print(log_fp, 'Not find reflection!')
                            success = False
                            break
                    else:
                        my_print(log_fp, 'unknow back frame number %d' % n_obb)
                        success = False
                        break
                elif label_l[2] == 'back_connector' or label_l[2] == 'back_support':
                    if n_obb == 1:
                        pass
                    elif n_obb == 2:
                        check, symm = reflect(obb_l)
                        if not check:
                            my_print(log_fp, '%s not reflectional' % k)
                            success = False
                            break
                        else:
                            syms.append(symm)
                    else:
                        check, symm = translation(obb_l)
                        if not check:
                            my_print(log_fp, '%s not translational' % k)
                            r_list = find_reflections_translation(obb_l)
                            if r_list:
                                syms.extend(r_list)
                            else:
                                my_print(log_fp, '%s find symmetry Fail!' % label_l[2])
                else:
                    my_print(log_fp, 'unknow back type: %s' % label_l[2])
                    success = False
                    break
            elif label_l[1] == 'chair_base':
                if label_l[3] == 'star_leg_set':
                    check, symm = rotation(obb_l)
                    if not check:
                        my_print(log_fp, 'Chair base Rotation error!')
                        success = False
                        break
                    else:
                        syms.append(symm)
                elif label_l[-1] == 'central_support' and n_obb > 1:
                    instances[k] = [merge(instances[k])]
                    n_obb = 1
                else:
                    if n_obb == 1:
                        pass
                    elif n_obb == 2:
                        my_print(log_fp, 'Two leg chair!')
                        check, ref = reflect(obb_l)
                        if check:
                            syms.append(ref)
                        else:
                            my_print(log_fp, '%s not reflectional' % k)
                            if label_l[1] != 'bar_stretcher':
                                success = False
                                break
                    elif n_obb == 3:#! can be translation
                        ref_list = find_reflections(obb_l, [0, 2, 1])
                        if ref_list:
                            syms.extend(ref_list)
                        else:
                            my_print(log_fp, 'Not find reflection!')
                            if label_l[1] != 'bar_stretcher':
                                success = False
                                break
                    else:
                        r_list = find_reflections(obb_l, [0, 2, 1])
                        if r_list:
                            syms.extend(r_list)
                        else:
                            my_print(log_fp, '4Leg reflection error!')
                            if label_l[1] != 'bar_stretcher':
                                success = False
                                break
                        # centroid = np.stack([o[:3] for o in obb_l])
                        # print(centroid)
                        # idx = np.argmin(centroid, 0)
                        # x_min = float(centroid[idx[0]][0])
                        # idx = np.argmax(centroid, 0)
                        # x_max = float(centroid[idx[0]][0])
                        # min_l = []
                        # max_l = []
                        # mid_l = []
                        # for j in range(n_obb):
                        #     if abs(centroid[j][0] - x_min) < 0.1:
                        #         min_l.append(j)
                        #     elif abs(centroid[j][0] - x_max) < 0.1:
                        #         max_l.append(j)
                        #     else:
                        #         mid_l.append(j)
                        # if len(min_l) == 2 and len(max_l) == 2:
                        #     syms.append(reflect(obb_l, *min_l))
                        #     syms.append(reflect(obb_l, *max_l))
                        # elif len(mid_l) == 2 and len(min_l) == 1 and len(max_l) == 1:
                        #     syms.append(reflect(obb_l, *mid_l))
                        #     syms.append(reflect(obb_l, min_l[0], max_l[0]))
                        # else:
                        #     my_print(log_fp, 'z axis')
                        #     idx = np.argmin(centroid, 0)
                        #     z_min = float(centroid[idx[0]][2])
                        #     idx = np.argmax(centroid, 0)
                        #     z_max = float(centroid[idx[0]][2])
                        #     min_l = []
                        #     max_l = []
                        #     mid_l = []
                        #     for j in range(n_obb):
                        #         if abs(centroid[j][2] - z_min) < 0.1:
                        #             min_l.append(j)
                        #         elif abs(centroid[j][2] - z_max) < 0.1:
                        #             max_l.append(j)
                        #         else:
                        #             mid_l.append(j)
                        #     if len(min_l) == 2 and len(max_l) == 2:
                        #         syms.append(reflect(obb_l, *min_l))
                        #         syms.append(reflect(obb_l, *max_l))
                        #     else:
                        #         my_print(log_fp, 'Leg reflection error!')
                        
                    # else:
                    #     centroid = np.stack([o[:3] for o in obb_l])
                    #     idx = np.argmin(centroid, 0)
                    #     z_min = float(centroid[idx[0]][2])
                    #     idx = np.argmax(centroid, 0)
                    #     z_max = float(centroid[idx[0]][2])
                    #     min_l = []
                    #     max_l = []
                    #     for j in range(n_obb):
                    #         if abs(centroid[j][2] - z_min) < 0.1:
                    #             min_l.append(j)
                    #         elif abs(centroid[j][2] - z_max) < 0.1:
                    #             max_l.append(j)
                    #         else:
                    #             my_print(log_fp, 'Leg in middle(multi leg)!')
                    #     check, symm = translation(obb_l, min_l)
                    #     if check:
                    #         syms.append(symm)
                    #     else:
                    #         my_print(log_fp, 'multi leg translation error!')
                    #         success = False
                    #         #break 
                    #     check, symm = translation(obb_l, max_l)
                    #     if check:
                    #         syms.append(symm)
                    #     else:
                    #         my_print(log_fp, 'multi leg translation error!')
                    #         success = False
                    #         #break 
                    # else:
                    #     my_print(log_fp, 'unknow leg number %s %d' % (label_l[-1], n_obb))
                    #     success = False
            elif label_l[1] == 'chair_arm':
                if n_obb == 1:
                    pass
                elif n_obb == 2:
                    check, ref = reflect(obb_l)
                    if check:
                        syms.append(ref)
                    else:
                        my_print(log_fp, '%s not reflectional' % k)
                        # success = False
                else:
                    r_list = find_reflections(obb_l, [0, 2, 1])
                    if r_list:
                        syms.extend(r_list)
                    else:
                        my_print(log_fp, '4 Arm reflection error!')
                        success = False
                        break
                # else:
                #     check, sym = translation(obb_l)
                #     if check:
                #         syms.append(sym)
                #     else:
                #         my_print(log_fp, 'Arm translation error!')
                #         success = False

        if label_l[1] == 'chair_back':
            labels += [0] * n_obb
        elif label_l[1] == 'chair_seat':
            labels += [1] * n_obb
        elif label_l[1] == 'chair_base':
            labels += [2] * n_obb
        elif label_l[1] == 'chair_arm':
            labels += [3] * n_obb
        else:
            labels += [-1] * n_obb
            my_print(log_fp, 'unknown label %s' % label_l[1])
            success = False
            break
        NOBBS += n_obb
        OBJS += instances[k]
    if success:
        adjs = adj_detcet_2([x['obb'] for x in OBJS], 0.06)
        save_obb('./new_obbs_3/%s.obb' % PATH.split('/')[-1], OBJS, adjs, syms, labels, 1)
    if 0 not in labels or 1 not in labels:
        success = False
    return success


if __name__ == "__main__":
    # PMAX = np.array([-9999, -9999, -9999])
    # PMIN = np.array([9999, 9999, 9999])
    
    log_fp = open('parser_log_3.txt', 'w')
    n_unsuccess = 0
    n_success = 0
    with open('../Chair.txt', 'r') as f:
        chair_list = f.read().strip().split('\n')
    for obj_name in chair_list:
        my_print(log_fp, '%d %s' % (n_unsuccess+n_success, obj_name))
        PATH = '../data_v0/' + obj_name
        NOBBS = 0
        OBJS = []
        try:
            success = main()
        except Exception as E:
            success = False
            my_print(log_fp, 'Exception!')
        if success:
            my_print(log_fp, 'Success')
            n_success += 1
        else:
            my_print(log_fp, 'Unsuccess')
            n_unsuccess += 1
    my_print(log_fp, 'Unsuccess: %d' % n_unsuccess)
    my_print(log_fp, 'Success: %d' % n_success)
    log_fp.close()