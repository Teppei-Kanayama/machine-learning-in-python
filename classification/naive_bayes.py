from _operator import add
from functools import reduce


class NaiveBayes(object):

    def __init__(self):
        self.pwc0 = {}
        self.pwc1 = {}
        self.pc0 = 0
        self.pc1 = 0
        self.vocaburary = None

    def fit(self, data0, data1):
        self.vocaburary = set(reduce(add, (map(lambda x: x.split(' '), (data0 + data1)))))
        self.pc0 = len(data0) / (len(data0) + len(data1))
        self.pc1 = len(data1) / (len(data0) + len(data1))

        for v in self.vocaburary:
            self.pwc0[v] = self._appear_count(data0, v) / len(data0)
            self.pwc1[v] = self._appear_count(data1, v) / len(data1)

    @staticmethod
    def _appear_count(data, v):
        return reduce(add, map(lambda sentence: v in sentence, data))

    def predict(self, test):
        prob0 = self.pc0
        prob1 = self.pc1
        for v in self.vocaburary:
            if v in test:
                prob0 *= self.pwc0[v]
                prob1 *= self.pwc1[v]

            else:
                prob0 *= 1 - self.pwc0[v]
                prob1 *= 1 - self.pwc1[v]
        return prob0, prob1


def main():
    data0 = [
        "good bad good good",
        "exciting exciting",
        "good good exciting boring",
    ]

    data1 = [
        "bad boring boring boring",
        "bad good bad",
        "bad bad boring exciting",
    ]

    test = 'good good bad boring'
    naive_bayes = NaiveBayes()
    naive_bayes.fit(data0, data1)
    prob0, prob1 = naive_bayes.predict(test)
    print(prob0, prob1)


if __name__ == '__main__':
    main()
