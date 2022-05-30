import numpy as np
import matplotlib.pyplot as plt

def calc_linear_bezier(p0, p1, t):
    """
    Calculates a linear bezier point parameterized by t by using the linear formula.
    Arguments:
        p0, p1: control point. Numpy array or list with coordinates in format [x, y]
        t: "t" value. Integer between 0 and 1 inclusive
    Returns:
        The coordinates of the calculated bezier point in format [x, y] and of numpy array type.
    """
    return np.dot(1-t, p0) + np.dot(t, p1)

def bezier_point_recursive(points, t):
    """
    Calculates a bezier point parameterized by t with a given set of control points.
    Arguments:
        points: control points. A numpy array with coordinates in format [x, y]
        t: "t" value. Integer between 0 and 1 inclusive
    Returns:
        The coordinates of the calculated bezier point in format [x, y] and of numpy array type.
    """
    new_points = np.empty([len(points) - 1, 2])
    
    for i in range(len(points) - 1):
        point = calc_linear_bezier(points[i], points[i + 1], t)
        new_points[i] = point
    
    if len(new_points) == 1:
        return new_points[0]
    
    return bezier_point_recursive(new_points, t)

def calc_cubic_bezier(p0, p1, p2, p3, t):
    """
    Calculates a cubic bezier point parameterized by t by using the cubic Bezier formula.
    Arguments:
        p0, p1, p2, p3: control point. Numpy array or list with coordinates in format [x, y]
        t: "t" value. Integer between 0 and 1 inclusive
    Returns:
        The coordinates of the calculated bezier point in format [x, y] and of numpy array type.
    """
    u = 1 - t
    return np.dot(pow(u, 3), p0) + np.dot(3 * pow(u, 2) * t, p1) + np.dot(3 * u * pow(t, 2), p2) + np.dot(pow(t, 3), p3)

def bezier_curve_equal_t_points(c_points, n):
    """
    Produces "n" bezier points by generating and evenly distributing "t" values. 
    Arguments:
        c_points: control points. Numpy array with coordinates in format [x, y]
        n: the total number of points to generate. An integer
    Returns:
        A numpy array with the coordinates of the calculated bezier points.
    """
    t_incr = 1 / (n - 1)
    t = 0
    bz_points = np.empty([n, 2])
    for i in range(n):
        interpolated_point = bezier_point_recursive(c_points, t)
        bz_points[i] = interpolated_point
        t += t_incr
    return bz_points

def calc_dist_between(x1, x2):
    """
    Calculates the distance between two numbers - x1 and x2.
    Arguments:
        x1: an integer
        x2: an integer
    Returns:
        An integer indicating the distance between x1 and x2.
    """
    x1_abs = abs(x1)
    x2_abs = abs(x2)

    if (x1 < 0 and x2 > 0) or (x2 < 0 and x1 > 0):
        return x1_abs + x2_abs

    if x1_abs > x2_abs:
        return x1_abs - x2_abs

    return x2_abs - x1_abs

def get_lut(c_points, lut_size):
    """
    Generates an array of pairs (or so-called LUT). Each pair consists of a distance and a "t" value.
    Arguments:
        c_points: control points. Numpy array with coordinates in format [x, y]
        lut_size: the total number of pairs to generate. An integer
    Returns:
        The LUT with the desired size. A numpy array.
    """
    arc_lengths = range(1, lut_size)
    
    multiplier = 1 / (lut_size - 1) # since the max t value is 1 devide it by the points to get the multiplier
    
    lut = np.empty([lut_size, 2])
    lut[0] = [0, 0]
    
    curr_curve_len = 0

    f = bezier_point_recursive(c_points, 0)
    for i in arc_lengths:
        t_next = i * multiplier
        f_next = bezier_point_recursive(c_points, t_next)
        
        dx = calc_dist_between(f[0], f_next[0])
        dy = calc_dist_between(f[1], f_next[1])
        hypotenuse_len = np.sqrt(dx * dx + dy * dy) # pythagorean theorem
        
        curr_curve_len += hypotenuse_len
        lut[i] = [curr_curve_len, t_next]
        f = f_next
    
    return lut

def distance_to_t(dist, lut):
    """
    Iterates through the provided LUT and finds the pair with the closest distance to the input.
    Arguments:
        dist: The desired distance that the returned "t" will represent. A float
        lut: The LUT that it will use to extract the "t" value. A numpy array
    Returns:
        An integer "t" value within the range of 0 and 1 inclusive.
    """
    n = len(lut) - 1
    arc_len = lut[len(lut) - 1][0]
    if (dist > arc_len): return 0
    
    for i in range(n):
        prev_dist = lut[i][0]
        next_dist = lut[i + 1][0]
        # if the distance is between the current 2 distances from the lut, return the closest distance t value
        if (dist >= prev_dist and dist <= next_dist):
            # figure out which one is closer
            close_to_prev = dist - prev_dist
            close_to_next = next_dist - dist
            if (close_to_prev <= close_to_next):
                return lut[i][1]
            else:
                return lut[i + 1][1]

def bezier_curve_equal_dist_points(c_points, number_bz_points, lut_size):
    """
    Produces "n" bezier points by generating and evenly distributing distance values. 
    Then converts the distance values to "t" values by using a LUT. 
    That way an arc length parameterization is accomplished.
    Arguments:
        c_points: control points. Numpy array with coordinates in format [x, y]
        number_bz_points: the total number of points to generate. An integer
        lut_size: the size (or precision) of the LUT that's going to be used. An integer
    Returns:
        A numpy array with the coordinates of the calculated bezier points.
    """
    lut = get_lut(c_points, lut_size)
    arc_length = lut[len(lut)-1][0]
    multiplier = arc_length / (number_bz_points - 1)
    bz_points = np.empty([number_bz_points, 2])

    for i in range(number_bz_points):
        dist = i * multiplier
        t = distance_to_t(dist, lut)
        interpolated_point = bezier_point_recursive(c_points, t)
        bz_points[i] = interpolated_point

    return bz_points

def plot_bezier_coordinates(control_points, bz_points, cp_lines = True, xlim = None, ylim = None, equal_aspect = False):
    """
    Plots a Bezier curve points and optionally the control points with the connecting lines.
    Arguments:
        c_points: control points. Numpy array with coordinates in format [x, y]
        bz_points: bezier points. Numpy array with coordinates in format [x, y]
        cp_lines: boolean defining if the control points will be rendered. A boolean with default value - True
        xlim: defines the x limits. Numpy array with coordinates in format [left, right]
        ylim: defines the y limits. Numpy array with coordinates in format [left, right]
        equal_aspect: boolean defining if the aspect ratio should be equal for x and y. A boolean with default value - False
    """
    x_bz = bz_points[:,0]
    y_bz = bz_points[:,1]
    plt.scatter(x_bz, y_bz, label="bz point")
    
    if cp_lines:
        x_control_point = control_points[:,0]
        y_control_point = control_points[:,1]
        plt.scatter(x_control_point, y_control_point, color="red", label="control point")
        plt.plot(x_control_point, y_control_point, linestyle="--", markersize=7, color="red")
    
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])

    if ylim != None:
        plt.ylim(ylim[0], ylim[1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()