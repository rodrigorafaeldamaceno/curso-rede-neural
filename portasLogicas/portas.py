def portaAnd(x, y):
    w1 = -1
    w2 = 1
    b = 0.5
    print("f(", x, ",", y, ") = ", x*w1+y*w2+b)

def portNot(x):
    w1 = -1
    b = 0.5
    print("Not: ", x*w1+b)

portNot(0)
portNot(1)


