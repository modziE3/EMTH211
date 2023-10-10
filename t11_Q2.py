import numpy as np
import matplotlib.pyplot as plt


def datas_getter():
    datas = []
    datas.append(np.genfromtxt('data_question_2a.csv', delimiter=","))
    datas.append(np.genfromtxt('data_question_2b.csv', delimiter=","))
    datas.append(np.genfromtxt('data_question_2c.csv', delimiter=","))
    return datas

def data_plotter(datas):
    for data in datas:
        x_data, y_data = data[:,0], data[:,1]
        fig, ax=plt.subplots() 
        ax.plot(x_data, y_data,'o')
        ax.axis("equal")
        plt.show()

def discrete_mean_values(datas):
    x_means = []
    y_means = []
    for data in datas: 
        x_data, y_data = data[:,0], data[:,1]
        x_means.append(np.mean(x_data))
        y_means.append(np.mean(y_data))
    return x_means, y_means

def correlation_finder(datas):
    correlations = []
    for data in datas:
        x_data, y_data = data[:,0], data[:,1]
        correlations.append(np.corrcoef(np.array(x_data), np.array(y_data)))
    return correlations


if __name__ == '__main__':
    datas = datas_getter()
    data_plotter(datas)
    x_means, y_means = discrete_mean_values(datas)
    correlations = correlation_finder(datas)
    data_id = ['A', 'B', 'C']
    for data_num in range(len(datas)): print(f"data {data_id[data_num]}: x mean{x_means[data_num]:.2f},   y mean {y_means[data_num]:.2f},   corr(x, y) = {correlations[data_num][0,1]:.2f}")