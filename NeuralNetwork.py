import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import pickle


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def softmax(x):
    return 1 / sum(np.exp(x)) * np.exp(x)


def cross_entropy_error(v, y):
    return -np.log(v[y])


data = h5py.File('data/MNISTdata.hdf5', 'r')
train_images = np.float64(data['x_train'][:])
train_labels = np.int64(np.array(data['y_train'][:, 0]))
test_images = np.float64(data['x_test'][:])
test_labels = np.int64(np.array(data['y_test'][:, 0]))
data.close()


class NeuralNetwork:
    first_layer = {}
    second_layer = {}

    def __init__(self, input_nodes, output_nodes, hidden_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes

        self.first_layer['para'] = np.random.randn(hidden_nodes, input_nodes) / np.sqrt(image_pixels)
        self.first_layer['bias'] = np.random.randn(hidden_nodes, 1) / np.sqrt(hidden_nodes)
        self.second_layer['para'] = np.random.randn(output_nodes, hidden_nodes) / np.sqrt(hidden_nodes)
        self.second_layer['bias'] = np.random.randn(output_nodes, 1) / np.sqrt(hidden_nodes)

    def forward(self, x, y):
        z = np.matmul(self.first_layer['para'], x).reshape((self.hidden_nodes, 1)) + self.first_layer['bias']
        h = np.array(activation_function(z)).reshape((self.hidden_nodes, 1))
        u = np.matmul(self.second_layer['para'], h).reshape((self.output_nodes, 1)) + self.second_layer['bias']
        predict_list = np.squeeze(softmax(u))
        error = cross_entropy_error(predict_list, y)

        dic = {
            'Z': z,
            'H': h,
            'U': u,
            'f_X': predict_list.reshape((1, self.output_nodes)),
            'error': error
        }
        return dic

    def back_propagation(self, x, y, f_result):
        e = np.array([0] * self.output_nodes).reshape((1, self.output_nodes))
        e[0][y] = 1
        du = (-(e - f_result['f_X'])).reshape((self.output_nodes, 1))
        db_2 = copy.copy(du)
        dc = np.matmul(du, f_result['H'].transpose())
        delta = np.matmul(self.second_layer['para'].transpose(), du)
        db_1 = delta.reshape(self.hidden_nodes, 1) * activation_function(f_result['Z']).reshape(self.hidden_nodes, 1)
        dw = np.matmul(db_1.reshape((self.hidden_nodes, 1)), x.reshape((1, 784)))

        grad = {
            'dC': dc,
            'db_2': db_2,
            'db_1': db_1,
            'dW': dw
        }
        return grad

    def optimize(self, b_result, l_rate):
        self.second_layer['para'] -= l_rate * b_result['dC']
        self.second_layer['bias'] -= l_rate * b_result['db_2']
        self.first_layer['bias'] -= l_rate * b_result['db_1']
        self.first_layer['para'] -= l_rate * b_result['dW']

    def loss(self, x_train, y_train):
        loss = 0
        for n in range(len(x_train)):
            y = y_train[n]
            x = x_train[n][:]
            loss += self.forward(x, y)['error']
        return loss

    def train(self, x_train, y_train, iterations, rate):
        rand_indices = np.random.choice(len(x_train), iterations, replace=True)

        def l_rate(base_rate, ite, iterations_number, schedule=False):
            if schedule:
                return base_rate * 10 ** (-np.floor(ite / iterations_number * 5))
            else:
                return base_rate

        count = 1
        loss_dict = {}
        test_dict = {}

        for i in rand_indices:
            f_result = self.forward(x_train[i], y_train[i])
            b_result = self.back_propagation(x_train[i], y_train[i], f_result)
            self.optimize(b_result, l_rate(rate, i, iterations, True))

            if count % 5000 == 0:
                if count % 15000 == 0:
                    loss = self.loss(x_train, y_train)
                    test = self.test(test_images, test_labels)
                    print('Trained for {} times'.format(count), 'loss = {}, accuracy = {}'.format(loss, test))
                    loss_dict[str(count)] = loss
                    test_dict[str(count)] = test
                else:
                    print('Trained for {} times'.format(count))
            count += 1

        print('Training finished!')
        return loss_dict, test_dict

    def test(self, x_test, y_test):
        total_correct = 0
        for n in range(len(x_test)):
            y = y_test[n]
            x = x_test[n][:]
            prediction = np.argmax(self.forward(x, y)['f_X'])
            if prediction == y:
                total_correct += 1
        return total_correct / np.float64(len(x_test))


num_iterations = 200000
learning_rate = 0.01
image_pixels = 28 * 28
num_outputs = 10
hidden_size = 300

"""
NN = NeuralNetwork(image_pixels, hidden_size, num_outputs)
cost_dict, tests_dict = NN.train(train_images, train_labels, num_iterations, learning_rate)
accuracy = NN.test(test_images, test_labels)

with open('data/save.txt', 'wb+') as f:
    pickle.dump(NN, f)
    pickle.dump(cost_dict, f)
    pickle.dump(tests_dict, f)


with open('data/save.txt', 'rb+') as t:
    for j in range(3):
        if j == 0:
            NN = pickle.load(t)
        elif j == 1:
            cost_dict = pickle.load(t)
        else:
            tests_dict = pickle.load(t)


plt.plot(cost_dict.keys(), cost_dict.values())
plt.ylabel('Loss function')
plt.xlabel('Number of iterations')
plt.xticks(rotation=60)
plt.title('Loss function')
plt.show()

plt.plot(tests_dict.keys(), tests_dict.values())
plt.ylabel('Test Accuracy')
plt.xlabel('Number of iterations')
plt.xticks(rotation=60)
plt.title('Test accuracy')
plt.show()
"""