import ANN
from random import randint
def main():

    trainingdataXOR = [0,0,0,0,1,1,1,0,1,1,1,0]
    nn = ANN.NN(2, 2, 1)
    for n in range(10000):
        r = randint(0,3)
        nn.train([trainingdataXOR[r],trainingdataXOR[r+1]], trainingdataXOR[r+2])

    Guess = nn.Predict([0,0])
    print(Guess)

    Guess = nn.Predict([0,1])
    print(Guess)
    Guess = nn.Predict([1,0])
    print(Guess)
    Guess = nn.Predict([1,1])
    print(Guess)


main()

