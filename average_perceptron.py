from collections import defaultdict
import codecs
from copy import deepcopy

# 0 B
# 1 M
# 2 E
# 3 S
class average_perceptron:

    def __init__(self):
        self.weights = defaultdict(int)
        self.deltas = []

    def argmax(self, features):
        assert len(features) == 4
        max = -1
        arg = -1
        for i in range(len(features)):
            tmp = self.multi(features[i])
            if tmp > max:
                arg = i
                max = tmp
        
        return arg

    def multi(self, dict):
        sum = 0.0

        for i in dict:
            sum += self.weights[i] * dict[i]
        
        return sum

    def train(self, feature_vec_vec, labels_vec_vec, iteration):

        for i in range(iteration):
            print 'iteration: ', i
            self._one_iteration(feature_vec_vec, labels_vec_vec)
        
        n = len(self.deltas)

        w0 = defaultdict(int)

        tmp = n
        for delta in self.deltas:     
            for i in delta:
                w0[i] += (1.0*tmp/n) * delta[i]
            tmp -= 1
        
        self.weights = w0


    
    def _one_iteration(self, feature_vec_vec, labels_vec_vec):
        assert len(feature_vec_vec) == len(labels_vec_vec)

        for i in range(len(feature_vec_vec)):
            feature_vec = feature_vec_vec[i]
            for j in range(len(feature_vec)):
                features = feature_vec[j]
                label = labels_vec_vec[i][j]

                predictType = self.argmax(features)
                # while predictType != label:
                #     self.update_weights(features[label], features[predictType])
                #     predictType = self.argmax(features)

                if label == predictType:
                    self.deltas.append(defaultdict(int))
                    continue
                else:
                    self.update_weights(features[label], features[predictType])

    def update_weights(self, right_feature, wrong_feature):

        for i in right_feature:
            self.weights[i] += right_feature[i]

        for i in wrong_feature:
            self.weights[i] -= wrong_feature[i]

        delta = deepcopy(right_feature)

        for i in wrong_feature:
            delta[i] -= wrong_feature[i]

        self.deltas.append(delta)

    
def getDataSet(input_path):

    file = codecs.open(input_path, 'r', 'utf-8')
    word_v_v = []
    label_v_v = []
    for line in file.readlines():
        print line
        word_list = line.strip().split()
        word_v = [u'B']
        label_v = [None]

        for word in word_list:
            if len(word) == 1:
                word_v.append(word)
                # 3 is S
                label_v.append(3)
            else:
                word_v.append(word[0])
                # 0 is B
                label_v.append(0)

                for w in word[1:len(word)-1]:
                    word_v.append(w)
                    # 1 is M
                    label_v.append(1)
                
                word_v.append(word[-1])
                # 2 is E
                label_v.append(2)
        word_v.append(u'E')
        label_v.append(None)

        word_v_v.append(word_v)
        label_v_v.append(label_v)

    return word_v_v, label_v_v

def getTrainFeature(train_word_v_v, train_label_v_v):
    assert len(train_word_v_v) == len(train_label_v_v)

    features_v_v = []
    label_v_v = []
    print 'len of train word v v is ', len(train_word_v_v)
    for i in range(len(train_word_v_v)):
        assert len(train_word_v_v[i]) == len(train_label_v_v[i])
        print i
        features_v = []
        label_v = []
        for j in range(1, len(train_word_v_v[i]) - 1):

            features = []

            for k in range(4):
                feature = defaultdict(int)

                f = train_word_v_v[i][j-1] + u'_' + str(k)
                feature[f] = 1

                f = train_word_v_v[i][j] + u'_' + str(k)
                feature[f] = 1

                f = train_word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = train_word_v_v[i][j-2] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(train_word_v_v[i]):
                #     f = train_word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1

                f = train_word_v_v[i][j-1] + u'_' + \
                    train_word_v_v[i][j] + u'_' + str(k)
                feature[f] = 1

                f = train_word_v_v[i][j] + u'_' + \
                    train_word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                f = train_word_v_v[i][j-1] + u'_' + \
                    train_word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = train_word_v_v[i][j-2] + u'_' + \
                #         train_word_v_v[i][j-1] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(train_word_v_v[i]):
                #     f = train_word_v_v[i][j+1] + u'_' + \
                #         train_word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1



                f = train_word_v_v[i][j-1] + u'_' + \
                    train_word_v_v[i][j] + u'_' + \
                    train_word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = train_word_v_v[i][j-2] + u'_' + \
                #         train_word_v_v[i][j-1] + u'_' + \
                #         train_word_v_v[i][j] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(train_word_v_v[i]):
                #     f = train_word_v_v[i][j] + u'_' + \
                #         train_word_v_v[i][j+1] + u'_' + \
                #         train_word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1
                
                features.append(feature)

            features_v.append(features)
            assert train_label_v_v[i][j] != None
            label_v.append(train_label_v_v[i][j])
        features_v_v.append(features_v)
        label_v_v.append(label_v)
    
    return features_v_v, label_v_v

def getTestSet(input_path):

    file = codecs.open(input_path, 'r', 'utf-8')
    word_v_v = []
    for line in file.readlines():
        word_list = line.strip().split()
        word_v = [u'B']
        for word in word_list:
            for i in word:
                word_v.append(i)
        word_v.append(u'E')
        word_v_v.append(word_v)
    return word_v_v

def getTestFeature(word_v_v):

    features_v_v = []

    for i in range(len(word_v_v)):
        features_v = []
        label_v = []
        for j in range(1, len(word_v_v[i])-1):

            features = []

            for k in range(4):
                feature = defaultdict(int)

                f = word_v_v[i][j-1] + u'_' + str(k)
                feature[f] = 1

                f = word_v_v[i][j] + u'_' + str(k)
                feature[f] = 1

                f = word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = word_v_v[i][j-2] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(word_v_v[i]):
                #     f = word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1

                f = word_v_v[i][j-1] + u'_' + \
                    word_v_v[i][j] + u'_' + str(k)
                feature[f] = 1

                f = word_v_v[i][j] + u'_' + \
                    word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                f = word_v_v[i][j-1] + u'_' + \
                    word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = word_v_v[i][j-2] + u'_' + \
                #         word_v_v[i][j-1] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(word_v_v[i]):
                #     f = word_v_v[i][j+1] + u'_' + \
                #         word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1


                f = word_v_v[i][j-1] + u'_' + \
                    word_v_v[i][j] + u'_' + \
                    word_v_v[i][j+1] + u'_' + str(k)
                feature[f] = 1

                # if j - 2 >= 0:
                #     f = word_v_v[i][j-2] + u'_' + \
                #         word_v_v[i][j-1] + u'_' + \
                #         word_v_v[i][j] + u'_' + str(k)
                #     feature[f] = 1
                # if j + 2 < len(word_v_v[i]):
                #     f = word_v_v[i][j] + u'_' + \
                #         word_v_v[i][j+1] + u'_' + \
                #         word_v_v[i][j+2] + u'_' + str(k)
                #     feature[f] = 1
                features.append(feature)

            features_v.append(features)
        features_v_v.append(features_v)
    
    return features_v_v


def predict(features_v_v, perceptron):

    res_label_v_v = []
    for i in range(len(features_v_v)):

        res_label_v = []
        for j in range(len(features_v_v[i])):

            res_label_v.append(perceptron.argmax(features_v_v[i][j]))
        
        res_label_v_v.append(res_label_v)
    
    return res_label_v_v

def writeBack(outputpath, word_v_v, label_v_v):

    file = codecs.open(outputpath, 'w', 'utf-8')

    for i in range(len(word_v_v)):

        for j in range(1, len(word_v_v[i]) - 2):
            file.write(word_v_v[i][j])

            if label_v_v[i][j-1] == 2 or label_v_v[i][j-1] == 3:
                file.write(u'  ')
        file.write(word_v_v[i][-2])
        file.write('\n')
    
    file.close()

            
    

if __name__ == '__main__':
    print 'reading train set'
    train_word_v_v, train_label_v_v = getDataSet('train_after_remove.txt')
    # train_word_v_v, train_label_v_v = getDataSet('mytrain.txt')

    # features_v_v actually is a three dimensional array
    # a features is feature[4]

    print 'get train feature'
    features_v_v, label_v_v = getTrainFeature(train_word_v_v, train_label_v_v)

    p = average_perceptron()

    print 'training'
    p.train(features_v_v, label_v_v, 10)

    print 'reading test set'
    word_v_v = getTestSet('test_after_remove.txt')
    # word_v_v = getTestSet('mytest.txt')

    print 'get test feature'
    testFeature_v_v = getTestFeature(word_v_v)
    print 'predicting'
    res_label_v_v = predict(testFeature_v_v, p)

    print 'write back'
    writeBack('moreFea10.txt', word_v_v, res_label_v_v)







