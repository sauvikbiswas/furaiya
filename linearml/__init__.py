# Copyright 2016, Sauvik Biswas 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stochastic Gradient Descent. Am I missing something? 
# w = w - alpha*Del Qi(w)

import numpy as np

#-------------- not working
def SGD(w, alpha, delQi, params):
    print alpha * delQi(w, params)
    #y = params['y']
    #x = params['x']
 
    #w = w + alpha * x
    w = w + alpha * delQi(w, params)
    return w

def diffn(w, params):
    y = params['y']
    x = params['x']
    h = w.transpose().dot(x)
    grad = (y-h) * x
    return grad

#--------------------------

def BGD(w, alpha, delQi, params):
    print delQi(w,params)
    w = w + alpha * delQi(w, params)
    return w

def lindQi(w, params):
    y = params['y']
    x = params['x']
    datalen, dummy = np.shape(y)
    h = x.dot(w)
    residue = (h-y)
    dQi = x.transpose().dot(residue)
    return (2.0/datalen)*dQi


init = np.ones((2,1))
lastw = init

points = np.genfromtxt('data.csv', delimiter=',')
y = points[:,1:]
x = np.hstack([np.ones((len(y), 1)), points[:,0:1]])

no_iter = 10

params = {
    'x': x,
    'y': y,
    }
for i in range(no_iter):
    w = BGD(lastw, 0.0001, lindQi, params)
    #print w
    lastw = w
    
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    print b_gradient*(N/2.0), m_gradient*(N/2.0)
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        #print b, m
    return [b, m]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 10
#    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    #print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

run()

#for i in range(len(y)):
#    params = {
#        'x': np.array([x[i]]).transpose(),
#        'y': y[i],
#        }
#
#    w = SGD(lastw, 0.05, diffn, params)
#    print w
#    lastw = w
