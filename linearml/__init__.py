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
    return -(2.0/datalen)*dQi


init = np.zeros((2,1))
lastw = init

points = np.genfromtxt('data.csv', delimiter=',')
y = points[:,1:]
x = np.hstack([np.ones((len(y), 1)), points[:,0:1]])

no_iter = 2

params = {
    'x': x,
    'y': y,
    }
for i in range(no_iter):
    w = BGD(lastw, 0.0001, lindQi, params)
    print w
    lastw = w
    
