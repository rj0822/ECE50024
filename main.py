# A re-implementation of "Learning with Noisy Labels (NIPS 2013)"
# ECE 50024 Machine Learning I - Course Project
# Acknowledgement: The following resources are referenced for the project
# (1) Original paper online: https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf
# (2) https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise
# (3) https://github.com/jamie2017/LearningWithNoisyLabels
# (4) https://paperswithcode.com/task/learning-with-noisy-labels

# How to run the experiment:
# (1) Create a folder named "Data" and a folder named "Plot" before running the program.
# (2) Define the size of dataset by manually change the n value in line 42.
# (3) Run the program.
# (4) Select the type of dataset. Type 1 for linearly separable dataset. Or type 2 for randomly distributed dataset.
# (5) Enter the noise rate for class with label -1.
# (6) Enter the noise rate for class with lable +1.
# (7) The data will be saved in the folder "Data" and the plot will be saved in the folder "Plot".


import collections
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


os.chdir(os.curdir)
os.chdir("..")


# Operation of the program
# Select the type of dataset (Type 1 for linearly separable dataset or type 2 for randomly distributed dataset)
data_type = float(input("Please select the type of synthetic data.\nType 1 for linearly separable dataset or type 2 for randomly distributed dataset: "))
if data_type == 1:
    is_random = False # Type 1 for linearly separable dataset
else:
    is_random = True # Type 2 for randomly distributed dataset
n = 2500 # n = Size of the dataset

# Enter the noise rate for label -1 and label +1
noise_rate_1 = float(input("Enter the noise rate for class label -1: "))
noise_rate_2 = float(input("Enter the noise rate for class label +1: "))


# Generate synthetic dataset
class GenerateData(object):
    def __init__(self, n):
        self.data_size = n
        self.label_map = collections.defaultdict(set)
    
    # Generate initial data (random numbers) and divide them into two classes for future labeling
    def generate_initial_data(self,n):
        range = 100 # range = Range of the data
        while len(self.label_map[1]) < n * 0.5:
            x = random.randrange(-range, range)
            y = random.randrange(-range, range)
            if x - 0.6 * range >= y:
                self.label_map[1].add((x, y))
        while len(self.label_map[-1]) < n * 0.5:
            x = random.randrange(-range, range)
            y = random.randrange(-range, range)
            if x <= y:
                self.label_map[-1].add((x, y))

    # Generate noise-free linearly separable dataset and define the corresponding labels
    def generate_separable_data(self):
        self.generate_initial_data(self.data_size)
        class_0 = [[-1, xy[0], xy[1]] for xy in self.label_map[-1]]
        class_1 = [[1, xy[0], xy[1]] for xy in self.label_map[1]]
        separable_data = class_0 + class_1
        print("(1) Noise-free linearly separable dataset generated.")
        return separable_data

    # Generate noise-free random dataset and define the corresponding labels
    def generate_random_data(self):
        xy, label = make_classification(n_samples=self.data_size, n_features=3, n_redundant=1, n_informative=2, n_clusters_per_class=2, flip_y=0.0001, weights=(0.5, 0.5))
        rand_data = []
        i1, i2 = 0, 0
        for j in zip(label, xy[:, 0], xy[:, 1]):
            if j[0] == 0:
                rand_data.append([-1, j[1], j[2]])
                i2 += 1
            else:
                rand_data.append(j)
                i1 += 1
        print("(2) Noise-free random dataset generated.")
        return rand_data, i1, i2

    # Add noise to the generated datasets
    def noise(self, data, p1, p2):
        noised_data = [list(j) for j in data]
        flip_label_0 = int(sum(1 for j in noised_data if j[0] == -1) * p1)
        flip_label_1 = int(sum(1 for j in noised_data if j[0] == 1) * p2)
        while flip_label_0 or flip_label_1:
            for i, j in enumerate(noised_data):
                if flip_label_0 == 0 and flip_label_1 == 0:
                    break
                random_flip = random.choice([-1, 1])
                if flip_label_1 == 0 and random_flip == 1:
                    ran_flip = -1
                elif flip_label_0 == 0 and random_flip == -1:
                    random_flip = 1
                if random_flip == j[0]:
                    noised_data[i][0] = - j[0]
                    if random_flip == 1:
                        flip_label_1 -= 1
                    elif random_flip == -1:
                        flip_label_0 -= 1
        print("(3) Noised dataset generated.")
        return noised_data

    # Split the dataset into training set and testing set
    def split_data(self, data):
        train_data, test_data = train_test_split(data, shuffle=True)
        print("(4) The dataset has been splited into train size and test size: {} vs {}".format(len(train_data), len(test_data)))
        return train_data, test_data


class TrainModel(object):
    # Obtain the data from GenerateData
    def __init__(self, data_size, is_random, p1, p2):
        self.read_data = GenerateData(data_size)
        self.true_data = {}
        if is_random:
            noise_free_data, self.n1, self.n2 = self.read_data.generate_random_data()
            self.make_random = True
        else:
            noise_free_data = self.read_data.generate_separable_data()
            self.i1 = self.i2 = self.read_data.data_size / 2
            self.make_random = False
        self.true_data_map(noise_free_data)
        noised_data = self.read_data.noise(noise_free_data, p1, p2)
        self.noised_train_set, self.noised_test_set = self.read_data.split_data(noised_data)
        self.noised_test_map = {(x, y): label for label, x, y in self.noised_test_set}
        self.predict_map = {}

    def true_data_map(self, data):
        for d in data:
            self.true_data[(d[1], d[2])] = d[0]

    # Train the model using SVM
    def train(self, train_set):
        train_x = [(j[1], j[2]) for j in train_set] # train_x = data
        train_y = [j[0] for j in train_set] # train_y = label
        classifier = svm.SVC()
        classifier.fit(train_x, train_y) # Use SVM to train the model
        test_x = [(j[1], j[2]) for j in self.noised_test_set] 
        predict_y = classifier.predict(test_x)
        self.predict_map = {(xy[0], xy[1]): int(label) for label, xy in zip(predict_y, test_x)}
        print("(5) Model successfully trained.")
        return classifier

    # Select the optimal solution using cross validation
    def select_decision(self, p1, p2):
        min_Rlf = float('inf')
        target_dataset = None
        data = np.array(self.noised_train_set)
        kf = KFold(n_splits=2)
        for train, test in kf.split(data):
            size = len(train)
            tr_data = data[train] # tr_data = True data
            x_t = [j[1] for j in tr_data ] # x_t = Data with true labels
            y_t = [j[2] for j in tr_data] # y_t = True labels
            label_f = [j[0] for j in tr_data] # label_f = Flipped labels (noise)
            label_t = [self.true_data[(x, y)] for x, y in zip(x_t, y_t)] # label_t = True labels
            pos = 0
            neg = 0
            for i in range(size):
                if label_f[i] == label_t[i]:
                    pos += 1
                else:
                    neg += 1
            p_y = 1.0 * sum(1 for j in tr_data if j[0] == -1) / size
            py = 1.0 * sum(1 for j in tr_data if j[0] == 1) / size
            Rlf = []
            for j in tr_data:
                Rlf.append(self.estimator(self.true_data[j[1], j[2]], j[0], py, p_y, p1, p2))
            if np.mean(Rlf) < min_Rlf:
                min_Rlf = np.mean(Rlf)
                target_dataset = tr_data
        print("(6) Cross validation completed.")
        return self.train(target_dataset)

    # Loss function for real-valued prediction: If prediction is correct, returns 0; if prediction is wrong, returns 1.
    def loss(self, fx, y):
        return 0 if fx == y else 1

    # Loss function for noisy dataset - Method of Unbiased Estimator (Lemma-1 in the paper)
    def estimator(self, x, y, py, p_y, p1, p2):
        return ((1 - p_y) * self.loss(x, y) - py * self.loss(x, -y)) / (1 - p1 - p2)

    # Accuracy calculation
    # Based on the paper, the accuracy is defined as the fraction of examples in the test set classified correctly w.r.t. the real-valued distribution
    def accuracy(self, prediction, acc):
        n_correct, n_total = 0, 0 # n_correct = Number of correct classfications; n_total = Total classifications
        for predict_class, real_class in zip(prediction, acc):
            n_total += 1
            if predict_class == real_class: # The classification is correct
                n_correct += 1
        acc = round(n_correct / n_total, 4)
        print("The Accuracy of the prediction is : {}".format(acc))
        return acc

    def comparison_plot(self, classifier, p1, p2, show_plot=True):

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5)) 

        COLOR = {-1: "r", 1: "k"} # Red dot represents Class = -1; black dot represents Class = +1

        # Plot 1: Noise-free data
        x_1 = [j[1] for j in self.noised_test_set]
        y_1 = [j[2] for j in self.noised_test_set]
        label_1 = [self.true_data[(x, y)] for x, y in zip(x_1, y_1)]
        color_1 = [COLOR[j] for j in label_1]
        ax1.scatter(x_1, y_1, marker='o', c=color_1, s=20)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Noise-free Data', font={'family':'Arial', 'size':15})
        

        # Plot 2: Data with noise
        x_2 = x_1
        y_2 = y_1
        label_2 = [j[0] for j in self.noised_test_set]
        color_2 = [COLOR[j] for j in label_2]
        ax2.scatter(x_2, y_2, marker='o', c=color_2, s=20)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Noisy Data\nNoise Rate = ' + str(p1) + " & " + str(p2), font={'family':'Arial', 'size':15})

        # Plot3 : Classification result
        x_3 = x_1
        y_3 = y_1
        result_x1 = zip(x_3, y_3)
        result_y1 = classifier.predict(list(result_x1))
        color_3 = [COLOR[j] for j in result_y1]
        acc = self.accuracy(label_1, result_y1)
        ax3.scatter(x_3, y_3, marker='o', c=color_3, s=20)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Classification Result\nAccuracy = ' + str(100*acc) + '%', font={'family':'Arial', 'size':15})

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        if show_plot:
            plt.show()

        self.save(f, p1, p2)
        return acc

    def save(self, f, p1, p2):
        table = []
        headers = ["Positions", "Noise-free data", "Noisy Data", "Unbiased_loss_predition"]
        for x, y in self.noised_test_map.keys():
            saved_data = (x, y), \
                    self.true_data[(x, y)], self.noised_test_map[(x, y)], \
                    self.predict_map[(x, y)]
            table.append(saved_data)
        file = pd.DataFrame(table, columns=headers)
        file.set_index("Positions")
        if self.make_random:
            filename = "Random" + str(self.read_data.data_size) + "-" + str(p1) + "-" + str(p2)
        else:
            filename = "Separable" + str(self.read_data.data_size) + "-" + str(p1) + "-" + str(p2)

        file.to_csv("Code/Data/" + filename + ".csv", index=False)
        print("Data saved as csv file.")
        f.savefig("Code/Plot/" + filename + ".png")
        print("Plot saved as png file.")


# Run the program
solve = TrainModel(n, is_random, noise_rate_1, noise_rate_2)
decision = solve.select_decision(noise_rate_1, noise_rate_2)
solve.comparison_plot(decision, noise_rate_1, noise_rate_2)

print("End of the program")