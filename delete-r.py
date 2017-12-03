import sys
import codecs
def remove_r(x):
    return x != u'\r' 

file = codecs.open(sys.argv[1], 'r', 'utf-8')

flst = file.readlines()

for i in range(len(flst)):
    flst[i] = filter(remove_r, flst[i])

outputFile = codecs.open(sys.argv[2], 'w', 'utf-8')

for i in flst:
    outputFile.write(i)


    