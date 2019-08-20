#!/usr/bin/env python3
# exercício: implementar o e-lógico


def f(x1, x2):
    v = x1 + x2 - 1.5
    y = passo(v)
    return y


def passo(v):
    return 1 if v >= 0 else 0


def main():
    dataset = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    for instancia in dataset:
        x1 = instancia[0]
        x2 = instancia[1]
        y = f(x1, x2)
        msg = "f(" + str(x1) + ", " + str(x2) + ") = " + str(y)
        print(msg)


main()
