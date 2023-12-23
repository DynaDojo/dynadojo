import math

import numpy as np

from ..abstractions import AbstractAlgorithm
from ..utils.seeding import temp_random_seed


class LowestPossibleRadius(AbstractAlgorithm):
    def __init__(self, embed_dim, timesteps, max_control_cost, seed):
        super().__init__(embed_dim, timesteps, max_control_cost, seed)
        self.currRadius = 1
        self.radiiTables = {self.currRadius: self.generateRadiusTable(self.currRadius)}
        self.ROW_LENGTH = embed_dim
        self._max_control_cost = max_control_cost

    def generateCombos(self, n):
        if not n:
            return

        for i in range(2**n):
            s = bin(i)[2:]
            s = "0" * (n-len(s)) + s
            yield s

    def generateRadiusTable(self, radius):
        allCombos = self.generateCombos((radius*2)+1)
        return dict.fromkeys(allCombos)

    def isValidRadius(self, radius, samples) -> bool:
        for sample in samples:
            for stepidx, step in enumerate(sample):
                if stepidx == 0:
                    continue

                for cellidx, cell in enumerate(step):
                    neighborhood = ""

                    # to the left
                    for neg_x in range(radius, 0, -1):
                        neighborhood += str(sample[stepidx - 1][cellidx - neg_x])

                    # directly above
                    neighborhood += str(sample[stepidx - 1][cellidx])

                    # to the right
                    for pos_x in range(1, radius+1):
                        pos = (cellidx + pos_x) % self.ROW_LENGTH
                        neighborhood += str(sample[stepidx - 1][pos])

                    if (self.radiiTables[radius][neighborhood]) == None:
                        self.radiiTables[radius][neighborhood] = cell
                    else:
                        if self.radiiTables[radius][neighborhood] != cell:
                            return False
        return True

    def fit(self, samples, **kwargs):
        while (not self.isValidRadius(self.currRadius, samples)):
            newRadius = self.currRadius+1
            self.radiiTables[newRadius] = self.generateRadiusTable(newRadius)
            self.currRadius = newRadius

    def act(self, x, **kwargs):
        with temp_random_seed(self._seed):
            control = []
            lastState = x[:, -1, :]
            control_mag = {}  # constraint is for each sample

            # find a smart control for 1st next state of traj
            for sampleidx, sample in enumerate(lastState):
                tempControl = []
                for _ in range(math.floor(self.embed_dim / ((self.currRadius * 2) + 1))):
                    cellidx = self.currRadius
                    neighborhood = ""

                    # left of cell
                    for neg_x in range(self.currRadius, 0, -1):
                        neighborhood += str(sample[cellidx - neg_x])

                    # the cell
                    neighborhood += str(sample[cellidx])

                    # right of cell
                    for pos_x in range(1, self.currRadius+1):
                        pos = (cellidx + pos_x) % self.ROW_LENGTH
                        neighborhood += str(sample[pos])

                    # if key is seen, replace with unseen
                    if (self.radiiTables[self.currRadius][neighborhood]) != None:
                        unseenKeys = [
                            key for key, value in self.radiiTables[self.currRadius].items() if value is None]
                        if unseenKeys:
                            desiredKey = np.random.choice(unseenKeys)

                            for idx, element in enumerate(neighborhood):
                                if sampleidx not in control_mag:
                                    control_mag[sampleidx] = 0
                                if control_mag[sampleidx] >= self._max_control_cost:
                                    tempControl.append(0)
                                else:
                                    if element == desiredKey[idx]:
                                        tempControl.append(0)
                                    elif element > desiredKey[idx]:
                                        tempControl.append(-1)
                                        control_mag[sampleidx] += 1
                                    elif element < desiredKey[idx]:
                                        tempControl.append(1)
                                        control_mag[sampleidx] += 1
                    cellidx += (self.currRadius*2)+1

                # finish the row of control if not done
                while (len(tempControl) < self.embed_dim):
                    tempControl.append(0)

                control.append([tempControl])

            # for all other states of traj, choose no control
            for sampleidx in range(len(x)):
                for _ in range(1, self._timesteps):
                    control[sampleidx].append(np.zeros(self.embed_dim))

            return np.array(control)

    def _evolve(self, x):
        evolved = []

        for sample in x:
            sampleResult = []
            for cellidx in range(0, self.ROW_LENGTH):
                neighborhood = ""

                # to the left
                for neg_x in range(self.currRadius, 0, -1):
                    neighborhood += str(sample[cellidx - neg_x])

                # directly above
                neighborhood += str(sample[cellidx])

                # to the right
                for pos_x in range(1, self.currRadius+1):
                    pos = (cellidx + pos_x) % self.ROW_LENGTH
                    neighborhood += str(sample[pos])

                # if seen before, use knowledge
                if (self.radiiTables[self.currRadius][neighborhood]) != None:
                    sampleResult.append(
                        self.radiiTables[self.currRadius][neighborhood])

                # else -> random guess between 0/1
                else:
                    sampleResult.append(np.random.randint(0, 1))
            evolved.append(sampleResult)
        return evolved

    def predict(self, x0, timesteps, **kwargs):
        preds = [x0]

        for _ in range(timesteps-1):
            preds.append(self._evolve(preds[-1]))

        preds = np.array(preds)
        preds = np.transpose(preds, (1, 0, 2))

        return preds
