import numpy as np
from evo import Evolution, Optimization
import matplotlib.pyplot as plt

def fun(x):
    return -x * (x - 1) * (x - 2) * (x - 3) * (x - 4)


def fitness(opt):
    return fun(opt.value[0])


evo = Evolution(
    pool_size=10, fitness=fitness, individual_class=Optimization, n_offsprings=3,
    pair_params={'alpha': 0.5},
    mutate_params={'lower_bound': 0, 'upper_bound': 4, 'rate': 0.25, 'dim': 1},
    init_params={'lower_bound': 0, 'upper_bound': 4, 'dim': 1}
)
n_epochs = 50

for i in range(n_epochs):
    evo.step()
    xs = np.arange(-0.2,4.1,1e-3)
    plt.plot(xs, [fun(x) for x in xs])
    values = [x.value[0] for x in evo.pool.individuals]
    plt.scatter(values, list(map(fun,values)),c='r')
    plt.show(block=False)
    plt.pause(0.5)
    print(len(values),end=' ')
    plt.clf()
print(evo.pool.individuals[-1].value)