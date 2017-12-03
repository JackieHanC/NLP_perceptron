from collections import defaultdict
import codecs

class CN_Seg_Perceptron:
    def __init__(self, activator):
        self.weights = defaultdict(float)
        self.activator = activator
        self.bias = 0

    def __str__(self):

        return 'weights\t%s\n' % self.weights


    # 1 represent divide,  0 represent not divide
    def predict(self, feature):
        return self.activator(
            self.dictMul(self.weights, feature)
            + self.bias
            )

    def dictMul(self, dict1, dict2):
        sum = 0
        for i in dict2:
            sum += dict1[i] * dict2[i]
        return sum

    def train(self, feature_vec_vec, labels_vec_vec, iteration, rate):
        for i in range(iteration):
            print 'iteration: ', i
            self._one_iteration(feature_vec_vec, labels_vec_vec, rate)

    def _one_iteration(self, feature_vec_vec, labels_vec_vec, rate):

        assert len(feature_vec_vec) == len(labels_vec_vec)

        for i in range(len(feature_vec_vec)):
            feature_vec = feature_vec_vec[i]
            labels_vec = labels_vec_vec[i]
            assert len(feature_vec) == len(labels_vec)

            for j in range(len(feature_vec)):
                
                feature = feature_vec[j]
                label = labels_vec[j]

                output = self.predict(feature)

                if output != label:

                    self.update_weights(feature, label, output, rate)

    
    def update_weights(self, feature, label, output, rate):

        delta = label - output

        for i in feature:
            self.weights[i] += rate * delta * feature[i]

        self.bias += rate * delta





def getDataset(train_data):
    file = codecs.open(train_data, 'r', 'utf-8')
    lst = file.readlines()

    train_vecs = []
    train_labels = []

    for line in lst:
        word_v = [u'B'] 
        label = [1]
        print line
        print 'len of this line', len(line)
        for j in range(len(line)):
            if line[j] == u'\n':
                break

            if line[j] == u' ':
                continue
            else:
                word_v.append(line[j])
                if j+1 >= len(line):
                    label.append(0)
                elif line[j+1] == u' ':
                    label.append(1)
                else:
                    label.append(0)

        word_v.append(u'E')
        label.append(1)

        train_vecs.append(word_v)
        train_labels.append(label)

    return train_vecs, train_labels

def getFeature(train_vecs, train_labels):
    assert len(train_vecs) == len(train_labels)

    feature_vec_vec = []

    labels_vec_vec = []
    for i in range(len(train_vecs)):
        assert len(train_vecs[i]) == len(train_labels[i])
        feature_vec = []
        labels_vec = []
        for j in range(1, len(train_vecs[i]) - 1):

            feature = defaultdict(float)

            f = train_vecs[i][j - 1]
            feature[f] = 1

            f = train_vecs[i][j]
            feature[f] = 1

            f = train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j]
            feature[f] = 1

            f = train_vecs[i][j] + u'_' + train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j] + u'_' \
                + train_vecs[i][j+1]
            feature[f] = 1

            feature_vec.append(feature)

            labels_vec.append(train_labels[i][j])

        
        feature_vec_vec.append(feature_vec)
        labels_vec_vec.append(labels_vec)
    
    return feature_vec_vec, labels_vec_vec


def getTestFeature(train_vecs):
    feature_vec_vec = []
    for i in range(len(train_vecs)):
        feature_vec = []
        
        for j in range(1, len(train_vecs[i]) - 1):

            feature = defaultdict(float)

            f = train_vecs[i][j - 1] 
            feature[f] = 1

            f = train_vecs[i][j]
            feature[f] = 1

            f = train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j]
            feature[f] = 1

            f = train_vecs[i][j] + u'_' + train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j+1]
            feature[f] = 1

            f = train_vecs[i][j-1] + u'_' + train_vecs[i][j] + u'_' \
                + train_vecs[i][j+1]
            feature[f] = 1

            feature_vec.append(feature)

            

        
        feature_vec_vec.append(feature_vec)
        
    
    return feature_vec_vec


def f(x):
    return 1 if x > 0 else 0

def getTestSet(test_data):
    file = codecs.open(test_data, 'r', 'utf-8')

    flst = file.readlines()

    word_vec_vec = []

    for line in flst:
        word_vec = [u'B']
        
        for i in range(len(line)):
            if line[i] != u'\n':
                word_vec.append(line[i])
        word_vec.append(u'E')

        word_vec_vec.append(word_vec)

    return word_vec_vec


def test(feature_vec_vec, perceptron):

    result_label_vec_vec = []
    
    for feature_vec in feature_vec_vec:
        result_label_vec = []
        for feature in feature_vec:
            ans = perceptron.predict(feature)
            result_label_vec.append(ans)
        result_label_vec_vec.append(result_label_vec)
    return result_label_vec_vec

def writeBack(output, word_vec_vec, labels_vec_vec):

    outputFile = codecs.open(output, 'w','utf-8')

    for i in range(len(word_vec_vec)):

        word_vec = word_vec_vec[i]
        for j in range(1, len(word_vec) - 1):
            outputFile.write(word_vec[j])

            if labels_vec_vec[i][j-1] == 1:
                outputFile.write(u'  ')
            
        if i+1 != len(word_vec_vec):
            outputFile.write(u'\n')
            
            





if __name__ == '__main__':
    train_vecs, train_labels = getDataset('train_after_remove.txt')
    # train_vecs, train_labels = getDataset('mytrain.txt')

    print 'get feature of train'
    feature_vec_vec, labels_vec_vec = getFeature(train_vecs, train_labels)
    print 'get feature of train done'


    p = CN_Seg_Perceptron(f)
    print 'start train'
    p.train(feature_vec_vec, labels_vec_vec, 6, 0.1)
    print 'end train'
    word_vec_vec = getTestSet('test_after_remove.txt')

    testFeature_vec_vec = getTestFeature(word_vec_vec)

    result_label_vec_vec = test(testFeature_vec_vec, p)

    writeBack('myAns.txt', word_vec_vec, result_label_vec_vec)










