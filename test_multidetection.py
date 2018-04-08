from multidetection import AnomalyDetector
import io, pstats, cProfile
import numpy as np

def gen_circle_data(N, r):
    """
    Generates random, circular data. Angles and radii are sampled from
    a uniform distribution.
    """
    thetas = np.random.uniform(-np.pi, np.pi, size=(N,1))
    rs = np.random.uniform(0, r, size=(N,1))
    x, y = rs * np.cos(thetas), rs * np.sin(thetas)
    return np.concatenate( [x, y], axis=1 )

# Generate some circles, and then overlay some random points
DEFAULT_NX = 1000
DEFAULT_NR = 200
x1_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([5, 5])
x2_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([-3, -1])
x3_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([2,1])
noise_gen  = lambda NR: np.concatenate( [np.random.uniform(-3,5,size=(NR,1)),
                                         np.random.uniform(-1,5,size=(NR,1))], axis=1 )
data_x = np.concatenate( [x1_gen(DEFAULT_NX), x2_gen(DEFAULT_NX),
                          x3_gen(DEFAULT_NX)] )
data = np.concatenate( [data_x, noise_gen(DEFAULT_NR)] )

box = AnomalyDetector()
pr = cProfile.Profile()
pr.enable()
box.fit( data_x )
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
