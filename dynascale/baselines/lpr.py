import numpy as np
from ..abstractions import Model



class LowestPossibleRadius(MyModel):
    def __init__(self, latent_dim, embed_dim, timesteps):
        super().__init__(latent_dim, embed_dim, timesteps)
        self.validRadii = [1]
        self.radiiTables = {}
        self.radiiTables[1] = self.generateRadiusTable(1)
        self.ROW_LENGTH = 0

    def generateCombos(self, n):
        if not n:
            return

        for i in range(2**n):
            s = bin(i)[2:]
            s = "0" * (n-len(s)) + s
            yield s

    def generateRadiusTable(self, radius):
        allCombos = self.generateCombos((radius*2)+1)
        tableDict = {}

        for combo in allCombos:
            tableDict[combo] = None

        return tableDict

    def train(self, samples, silent = False):
        #print(samples)
        self.validRadii = [1]
        self.radiiTables = {}
        self.radiiTables[1] = self.generateRadiusTable(1)
        self.ROW_LENGTH = len(samples[0][0])
        

        for sample in samples:
            for stepidx, step in enumerate(sample):
                if stepidx == 0:
                    continue

                for cellidx, cell in enumerate(step):
                    for radius in self.validRadii:
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

                        if(self.radiiTables[radius][neighborhood]) == None:
                            self.radiiTables[radius][neighborhood] = cell
                        else:
                            if self.radiiTables[radius][neighborhood] != cell:
                                if not silent:
                                    print("knockout of radius:" + str(radius))
                                # add a radius that is 1 larger
                                newRadius = self.validRadii[-1]+1
                                self.validRadii.remove(radius)
                                self.validRadii.append(newRadius)
                                self.radiiTables[newRadius] = self.generateRadiusTable(newRadius) 
        
    def act(self, x, **kwargs):
        radius = self.validRadii[-1]

        control = []

        for sample in samples:
            for stepidx, step in enumerate(sample):
                if stepidx == len(sample) - 1:
                    continue

                for y in range(self.latent_dim / ((radius*2) + 1)):
                    for radius in self.validRadii:
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

                        if(self.radiiTables[radius][neighborhood]) != None:
                            emptyKeys = [key for key, value in self.radiiTables[radius].items() if value is None]
                            desiredKey = random.choice(emptyKeys)

                             # edit neighborhood to be this 
                            for idx, element in enumerate(neighborhood):
                                if element == desiredKey[idx]:
                                    control += 0
                                elif element > desiredKey[idx]:
                                    control += -1
                                elif element < desiredKey[idx]:
                                    control += 1
                           

       
    
    def predict(self, x0, timesteps, **kwargs):
        result = []
        # take the only still possible radius
        predictedRadius = self.validRadii[-1]

        for cellidx in range(0, self.ROW_LENGTH):
            neighborhood = ""

            # to the left
            for neg_x in range(predictedRadius, 0, -1):
                neighborhood += str(sample[cellidx - neg_x])

            # directly above
            neighborhood += str(sample[cellidx])

            # to the right 
            for pos_x in range(1, predictedRadius+1):
                pos = (cellidx + pos_x) % self.ROW_LENGTH
                neighborhood += str(sample[pos])

            # if seen before
            if(self.radiiTables[predictedRadius][neighborhood]) != None:
                result.append(self.radiiTables[predictedRadius][neighborhood])

            # else -> random guess between 0/1
            else:
                result.append(random.randint(0, 1))

        return result