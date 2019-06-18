import numpy as np
import copy
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# Generate training set
# 100 training sets, each consisting of 10 sequences
def training_data_generate(num_training_set = 100, num_sequence = 10):
    random.seed(27)
    training_data = [[None] * num_training_set] * num_sequence
    initial_state = 3
    training_data = [[None] * num_training_set] * num_sequence
    for i in range(num_sequence):
        for j in range(num_training_set):
            x_vectors = np.zeros(7)
            step_now = initial_state
            x_vectors[step_now] = 1
            while step_now not in [0,6]:
                x_vector = np.zeros(7)
                if np.random.uniform() < 0.5:
                    step_now += 1
                else:
                    step_now += -1
                x_vector[step_now] = 1
                x_vectors = np.vstack([x_vectors,x_vector])
            # if step_now is in the absorbing state then while loop ends
                training_data[i][j] = x_vectors
    return training_data
# print training_data[0][1]


# Calculate delta weights
def delta_weights_calculate(training_sequence, alpha, lambda_, initial_weights = [0.5, 0.5, 0.5, 0.5, 0.5]):
    #print initial_weights
    delta_weights = 0
    sequence_non_terminal = training_sequence[:,1:-1]
    #print sequence_non_terminal.shape
    #print training_sequence
    if np.argmax(training_sequence[-1]) == 0:
        z = 0
    else :
        z = 1
    #print z

    t = training_sequence.shape[0] - 1
    for k in range(t):
        #print '~~~'
        #print 'initial_weights: ', initial_weights
        # print '---'
        if k == t-1:
            # terminal state
            delta_p = z - np.sum(np.multiply(np.array(initial_weights), sequence_non_terminal[k]))
            # print '---'
            # print 'initial_weights: ', np.array(initial_weights)
            # print 'sequence: ', sequence_non_terminal[k-1]
            # print 'delta_p(terminal state): ', delta_p
        else:
            # non-terminal state
            # print '---'
            delta_p = np.sum(np.multiply(sequence_non_terminal[k+1], np.array(initial_weights))) -\
                      np.sum(np.multiply(sequence_non_terminal[k], np.array(initial_weights)))
            #print '---'
            #print np.array(initial_weights)
            # print 'delta_p(non-terminal state): ', delta_p
        lambda_array = np.array([lambda_**(k-i) for i in range(k+1)])
        gradient = np.sum(np.transpose(sequence_non_terminal[:k+1]) * lambda_array, axis=1)

        # print 'gradient: ', gradient
        # print 'delta_p: ', delta_p
        # print 'gradient: ', gradient
        delta_weights += alpha * delta_p * gradient
    return delta_weights

def sutton_experiment_1(training_data, alpha, lambda_, initial_weights = [0.5, 0.5, 0.5, 0.5, 0.5],
                        target_weights  = [1/6.0,1/3.0,1/2.0,2/3.0,5/6.0], convergence_level=0.0001):
    # update weights after each sequence
    num_training_set = len(training_data[0])
    num_sequence = len(training_data)

    start_weights = [0.0, 0.0, 0.0, 0.0,0.0]
    while abs(np.max(np.array(initial_weights)-np.array(start_weights))) > convergence_level:
        start_weights = copy.copy(initial_weights)
        for x in range(num_sequence):
            delta_w = np.zeros(5)
            for y in range(num_training_set):
                delta_w += delta_weights_calculate(training_data[x][y], alpha, lambda_, initial_weights)
            #print 'delta_w: ', delta_w
            initial_weights += delta_w
        #print abs(np.max(np.array(initial_weights)-np.array(start_weights)))
        #print i, ': ', initial_weights
    #RMSE = (np.sum(np.array(target_weights) - np.array(initial_weights))**2/5)**0.5
    RMSE = (np.sum((target_weights - initial_weights) ** 2) / 5) ** 0.5
    return RMSE

def sutton_experiment_2(training_data, alpha, lambda_, initial_weights = [0.5, 0.5, 0.5, 0.5, 0.5],
                        target_weights=[1 / 6.0, 1 / 3.0, 1 / 2.0, 2 / 3.0, 5 / 6.0]):
    RMSE = 0
    num_training_set = len(training_data[0])
    num_sequence = len(training_data)
    for x in range(num_sequence):
        initial_weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        for y in range(num_training_set):
            delta_w = delta_weights_calculate(training_data[x][y], alpha, lambda_, initial_weights)
            initial_weights += delta_w
            RMSE += (np.sum((target_weights - initial_weights) ** 2) / 5) ** 0.5
    return RMSE/(num_sequence * num_training_set)




# #########################################################################################################
# print "Start to generate all charts in the report!"
# # Figure 3
# training_data = training_data_generate(100, 10)
# lambda_array_1 = np.linspace(0,1,20)
# #lambda_array_1 = np.array([0.2])
# RMSE_array_1 = []
# for i in range(len(lambda_array_1)):
#     lambda_ = lambda_array_1[i]
#     #print 'Training lambda = ', lambda_
#     RMSE_array_1.append(sutton_experiment_1(training_data, alpha=0.001, lambda_=lambda_))
# # print RMSE_array_1
# # Plot Figure 3
# plt.figure(figsize=(16,9))
# plt.plot(lambda_array_1, RMSE_array_1, '-o')
# plt.ylabel('RMSE')
# plt.xlabel(r'$\lambda$')
# plt.title('Figure 3 Replication (Convergence level = 0.0001)')
# plt.show()
#
# RMSE_array_1_1 = []
# for i in range(len(lambda_array_1)):
#     lambda_ = lambda_array_1[i]
#     #print 'Training lambda = ', lambda_
#     RMSE_array_1_1.append(sutton_experiment_1(training_data, alpha=0.001, lambda_=lambda_,convergence_level=0.1))
# # print RMSE_array_1
# # Plot Figure 3
# plt.figure(figsize=(16,9))
# plt.plot(lambda_array_1, RMSE_array_1_1, '-o')
# plt.ylabel('RMSE')
# plt.xlabel(r'$\lambda$')
# plt.title('Figure 3 Replication (Convergence level = 0.1)')
# plt.show()



#########################################################################################################
# Figure 4
training_data = training_data_generate(10, 100)
num_alpha = 10
lambda_array_2 = np.array([0.0, 0.3, 0.8, 1.0])
alpha_array = np.linspace(0,0.8,num_alpha)
RMSE_array_2 = np.zeros([4,num_alpha])

for i in range(4):
    lambda_ = lambda_array_2[i]
    for j in range(num_alpha):
        alpha = alpha_array[j]
        RMSE_array_2[i][j] = sutton_experiment_2(training_data, alpha, lambda_)
#print type(RMSE_array_2)
# Plot Figure 4

x_2 = alpha_array
plt.figure(figsize=(16,9))

plt.plot(x_2, RMSE_array_2[0,:], '-o', label=r'$\lambda = 0$')
plt.plot(x_2, RMSE_array_2[1,:], '-o', label=r'$\lambda = 0.3$')
plt.plot(x_2, RMSE_array_2[2,:], '-o', label=r'$\lambda = 0.8$')
plt.plot(x_2, RMSE_array_2[3,:], '-o', label=r'$\lambda = 1.0$')
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
plt.legend()
plt.ylabel('RMSE')
plt.ylim(0, 1.0)
plt.xlabel(r'$\alpha$')
plt.title('Figure 4 Replication')
plt.show()
#
#
#
# #########################################################################################################
# # Figure 5
# training_data = training_data_generate(10, 100)
# alpha_best = 0.3
# lambda_array_3 = np.linspace(0,0.8,20)
# RMSE_array_3 = []
# for i in range(len(lambda_array_3)):
#     lambda_ = lambda_array_3[i]
#     RMSE_array_3.append(sutton_experiment_2(training_data, alpha_best, lambda_))
# # print RMSE_array_3
#
# # Plot Figure 5
# plt.figure(figsize=(16,9))
# plt.plot(lambda_array_3, RMSE_array_3, '-o')
# plt.ylabel(r'RMSE with best $\alpha = 0.3$')
# plt.ylim(0.1,0.2)
# plt.xlabel(r'$\lambda$')
# plt.title(r'Figure 5 Replication with $\alpha = 0.3$')
# plt.show()
#
# alpha_best = 0.1
# lambda_array_3 = np.linspace(0,0.8,20)
# RMSE_array_3_1 = []
# for i in range(len(lambda_array_3)):
#     lambda_ = lambda_array_3[i]
#     RMSE_array_3_1.append(sutton_experiment_2(training_data, alpha_best, lambda_))
# # print RMSE_array_3
#
# # Plot Figure 5
# plt.figure(figsize=(16,9))
# plt.plot(lambda_array_3, RMSE_array_3_1, '-o')
# plt.ylabel(r'RMSE with best $\alpha = 0.1$')
# plt.ylim(0.1,0.2)
# plt.xlabel(r'$\lambda$')
# plt.title(r'Figure 5 Replication with $\alpha = 0.1$')
# plt.show()

print "All charts in the report has been generated!"