import ANN
from random import randint
def main():
    nn = ANN.NN(2, 2, 1)
    for n in range (250):
          nn.train([0,0],0)
    Guess = nn.Predict([0,0])
    print(Guess)
    for n in range (250):
          nn.train([0,1],1)
    Guess = nn.Predict([0,1])
    print(Guess)
    for n in range (250):
          nn.train([1,1],0)
    Guess = nn.Predict([1,1])
    print(Guess)
    for n in range (250):
          nn.train([1,0],1)
    Guess = nn.Predict([1,0])
    print(Guess)



main()

