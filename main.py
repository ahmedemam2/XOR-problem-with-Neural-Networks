import ANN
import random
def main():

    nn = ANN.NN(2, 2, 1)
    for n in range(5000):

        nn.train([1,0], 3)

    Guess = nn.Guess([1,0])
    print(Guess)



main()

