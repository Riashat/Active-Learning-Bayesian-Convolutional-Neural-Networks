This is the code for the implementation of black-box alpha according to the
paper:

Hernández-Lobato J. M., Li Y., Rowland  M., Bui T. D., Hernández-Lobato D. and
Turner R. E.  Black-Box Alpha Divergence Minimization, In ICML, 2016

The folder "boston_housing" contains code to train a neural network with one
hidden layer with 100 units on the Boston Housing dataset. There are two
versions of the code, one in in autograd in the folder "autograd". There is
another version in thenano in the folder "theano". The theano version can be
run in a GPU by typing

$ cd boston_housing/theano/ 
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python experiment.py 

If you want to run the code on the cpu you can just replace the previous line
with

$ python experiment.py 

The autograd version of the code is executed by running

$ cd boston_housing/autograd/ 
$ python experiment.py

The folder "mnist" contains theano code to run black-box alpha for training
a neural network with two hidden layers with 400 units each on the mnist
dataset. It is recommended to run this code in a gpu, otherwise it can be very
slow. This can be done by typing

$ cd mnist/
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python experiment.py 

The scripts experiment.py store the results in the folder "results".

All the scripts here use the value alpha = 0.5. This can be changed by
modifying the lines 

44 in boston_housing/autograd/black_box_alpha.py
93 in boston_housing/theano/experiment.py
117 in mnist/experiment.py    
